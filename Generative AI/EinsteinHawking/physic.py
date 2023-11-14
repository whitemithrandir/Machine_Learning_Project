import os
from Albert import Einstein 
from Stephen import Hawking 

#NOTE albert için token ayarlamsı yap
if __name__ == "__main__":

    instance = Einstein()
    model_path="/home/sabankara/coding/physics/physic-training-local1"


    if len(os.listdir(model_path)) == 0:

        dataset = instance.preprocessing_and_data()
        original_model, tokenizer = instance.huggingface_model_load()
        tokenized_datasets = dataset.map(instance.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['role_1', 'topic;', 'sub_topic', 'message_1','message_2'])
        peft_model = instance.setup_peft_model(original_model)
        fine_tune_trainer = instance.finetune_training_model(peft_model, tokenized_datasets)
        instance.save_model(fine_tune_trainer, model_path)
        

    else:
        dataset = instance.preprocessing_and_data()
        _ , tokenizer = instance.huggingface_model_load()
        trained_model = instance.load_model(model_path)

        example_indices_full = 28
        example_index_to_summarize = 279

        one_shot_prompt = instance.make_prompt(example_indices_full, example_index_to_summarize, dataset)

        Einstein_response = instance.model_output(trained_model,tokenizer, one_shot_prompt)
        print()
        print(Einstein_response)
        print()