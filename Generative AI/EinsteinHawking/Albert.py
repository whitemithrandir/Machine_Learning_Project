from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration
import torch
import time
import evaluate
import pandas as pd
import numpy as np
import psutil
import datasets
import warnings
import os
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig
import swifter
from dotenv import load_dotenv
load_dotenv()

class Einstein:
    def __init__(self):
        pass

    def huggingface_model_load(self):
        model_name=os.getenv("model_name")

        original_model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map= {"":0})
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # self.print_number_of_trainable_model_parameters(original_model)
        self.tokenizer = tokenizer
        return original_model, tokenizer

    def print_number_of_trainable_model_parameters(self, model):
    #NOTE model = self.original_model
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        print(f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")

    def preprocessing_and_data(self,tokenizer):
        huggingface_dataset_name = os.getenv("huggingface_dataset_name")
        dataset = load_dataset(huggingface_dataset_name)

        ##########################
        df_pandas = pd.DataFrame(dataset["train"])
        # print(df_pandas["topic;"].unique())
        df_pandas = df_pandas[df_pandas["topic;"].isin(['Special relativity','Dark matter','Black holes','Quantum mechanics','Plasma physics','Particle physics','String theory','Nuclear physics','Atomic physics','Quantum field theory','Gravitational waves','Electromagnetism','Chaos theory'])]
        df_pandas.reset_index(inplace=True, drop=True)
        df_pandas = df_pandas.drop(['role_1', 'topic;','sub_topic'], axis=1)
        df_pandas.rename(columns={'message_1': 'Question_1', 'message_2': 'Answer_1'}, inplace=True)

        df_pandas['Answer_2'] = df_pandas['Answer_1'].shift(-1)
        df_pandas['Question_2'] = df_pandas['Question_1'].shift(-1)
        df_pandas.dropna(inplace=True)

        df_pandas= df_pandas[["Answer_1","Question_1","Answer_2","Question_2"]]

        # tokenize process
        tokenized_df = df_pandas.applymap(lambda x: tokenizer(x, return_tensors='pt'))



        #NOTE instruction_prompt = 28 | Answer_1 = 150 |Question_1(mean) = 50
        q1_series = tokenized_df["Question_1"].swifter.apply(lambda x: tokenizer.decode(x["input_ids"][0][0:50],skip_special_tokens=True))
        a1_series = tokenized_df["Answer_1"].swifter.apply(lambda x: tokenizer.decode(x["input_ids"][0][0:370],skip_special_tokens=True))
        q2_series = tokenized_df["Question_2"].swifter.apply(lambda x: tokenizer.decode(x["input_ids"][0][0:50]))


        q1_df = pd.DataFrame(q1_series)
        a1_df = pd.DataFrame(a1_series)
        q2_df = pd.DataFrame(q2_series)

        df_pandas_new = pd.concat([q1_df , a1_df, q2_df, df_pandas["Answer_2"]], axis=1)

        df_pandas_new.sample(frac=1, random_state=42).reset_index(inplace=True, drop=True)


        #pandas to DatasetDict
        dataset1 = Dataset.from_dict(df_pandas_new)
        dataset = datasets.DatasetDict({"train":dataset1})

        train_testvalid = dataset['train'].train_test_split(test_size=0.05)

        test_valid = train_testvalid['test'].train_test_split(test_size=0.05)

        dataset = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'validation': test_valid['train']})

        return dataset



    #  inputs_id  label
    #  A1 Q1 A2     Q2

    def tokenize_function(self,example):
        start_prompt = 'Answer the following question:\n\n'
        instruction_prompt= start_prompt+"Question\n\n{Q1}\n\nAnswer\n\n{A1}\nQuestion\n\n{Q2}\n\nAnswer\n\n"
        prompt = [instruction_prompt.format(Q1=q1, A1=a1, Q2=q2) for q1, a1, q2 in zip(example["Question_1"], example["Answer_1"], example["Question_2"])]
    
        example['input_ids'] = self.tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda(0)

        example['labels'] = self.tokenizer(example["Answer_2"], padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda(0)

        return example


    def tokenize_setup(self, dataset):

        # The dataset actually contains 3 diff splits: train, validation, test.
        # The tokenize_function code is handling all data across all splits in batches.
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['Question_1', 'Answer_1', 'Question_2', 'Answer_2',])
        # tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)
        return tokenized_datasets



    def setup_peft_model(self, original_model):
        

        lora_config = LoraConfig(
            r=128, # Rank
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="all",
            task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
        )

        peft_model = get_peft_model(original_model, 
                                    lora_config)
        self.print_number_of_trainable_model_parameters(peft_model)

        return peft_model
    
    def finetune_training_model(self, peft_model, tokenized_datasets ):
        output_dir = f'/home/sabankara/coding/GenerativeAIforNLP/physic-training-{str(int(time.time()))}'

        peft_training_args = TrainingArguments(
            output_dir=output_dir,
            report_to="none",
            auto_find_batch_size=True,
            learning_rate=5e-3, # Higher learning rate than full fine-tuning.
            num_train_epochs=50,
            logging_steps=1,
            max_steps=400   
        )


        peft_trainer = Trainer(
            model= peft_model,
            args=peft_training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"]
        )

        peft_trainer.train()

        return peft_trainer
    
    def save_model(self, peft_trainer, model_path, tokenizer):
        
        peft_trainer.model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)   

    def load_model(self, model_path):
        peft_model_base = T5ForConditionalGeneration.from_pretrained(os.getenv("model_name"), torch_dtype=torch.bfloat16, device_map= {"":0})

        peft_model = PeftModel.from_pretrained(peft_model_base, 
                                            model_path, 
                                            torch_dtype=torch.bfloat16,
                                            is_trainable=False)
        return peft_model
    
    

    def zero_shot_make_prompt(example_indices):
        dataset_df = pd.DataFrame(dataset["test"])

        start_prompt = 'Answer the following question:\n\n'
        instruction_prompt= start_prompt+"Question\n\n{Q1}\n\nAnswer\n\n"
        #Answer1\n\n
        
        question = dataset_df['Question_1'][example_indices]
    #     answer = df_pandas["Answer_1"][example_indices]
        one_shot_prompt = instruction_prompt.format(Q1=question)


        return one_shot_prompt
    
 
# if __name__ == "__main__":

#     instance = Einstein()

#     model_path="/home/sabankara/coding/physics/physic-training-local1"


    # if len(os.listdir(model_path)) == 0:

    #     dataset = instance.preprocessing_and_data()
    #     original_model, tokenizer = instance.huggingface_model_load()
    #     tokenized_datasets = dataset.map(instance.tokenize_function, batched=True)
    #     tokenized_datasets = tokenized_datasets.remove_columns(['role_1', 'topic;', 'sub_topic', 'message_1','message_2'])
    #     peft_model = instance.setup_peft_model(original_model)
    #     fine_tune_trainer = instance.finetune_training_model(peft_model, tokenized_datasets)
    #     instance.save_model(fine_tune_trainer, model_path,tokenizer)
        

#     else:
#         dataset = instance.preprocessing_and_data()
#         _ , tokenizer = instance.huggingface_model_load()
#         trained_model = instance.load_model(model_path)

#         example_indices_full = 167
#         example_index_to_summarize = 246

#         one_shot_prompt = instance.make_prompt(example_indices_full, example_index_to_summarize, dataset)

#         Einstein_response = instance.model_output(trained_model,tokenizer, one_shot_prompt)
#         print()
#         print(Einstein_response)
    