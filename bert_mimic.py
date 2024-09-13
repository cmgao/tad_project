from transformers import AutoModel, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_df(file_path):
    data = pd.read_csv(file_path, sep=',', header=None, names=)
    
    return data



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, help='Input corpus')

    args=parser.parse_args()
    data = load_df(args.input_csv)

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
    
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    
    def summarize(input_text, batch_size=32):
        
        outputs = []
        total_batches = len(input_text) // batch_size
        
        for i in tqdm(range(0, len(input_text)), total=len(input_text), desc='summarizing'):
            
            inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
            generate_ids = model.generate(**inputs, max_new_tokens=260)
            batch_outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            result = [batch_outputs[-1].split(f'{args.output_lang}:')[-1].strip()]
            outputs.extend(result)
            
        return outputs
    

if __name__ == '__main__':
    main()