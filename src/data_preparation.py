#!/usr/bin/env python3
"""
Social media text data preparation pipeline with domain adaptation.

Key Features:
1. Twitter-specific text cleaning
2. MSMARCO dataset processing
3. SBERT fine-tuning data generation

Time Complexity:
- O(n) for text cleaning
- O(n log n) for data splitting
"""

import os
import re
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDING_DIR = DATA_DIR / "embeddings"

class DataPreprocessor:
    """Handle social media text cleaning and preparation"""
    
    def __init__(self):
        self.clean_patterns = [
            (r'@\w+', ''),  # Remove mentions
            (r'http\S+', ''),  # Remove URLs
            (r'#(\w+)', r'\1'),  # Keep hashtag text
            (r'[^\w\s]', ' '),  # Replace special chars
        ]

    def clean_text(self, text: str) -> str:
        """Apply Twitter-specific cleaning rules"""
        for pattern, repl in self.clean_patterns:
            text = re.sub(pattern, repl, text)
        return text.strip()

    def process_msmarco(self, df) -> Path:
        """Process MSMARCO dataset with domain adaptation"""
        print("Processing MSMARCO data from HuggingFace")
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Train/val/test split (70/15/15)
        train, temp = train_test_split(df, test_size=0.3)
        val, test = train_test_split(temp, test_size=0.5)
        
        # Save processed data
        output_path = PROCESSED_DIR / "msmarco.json"
        with open(output_path, 'w') as f:
            json.dump({
                'train': train['cleaned_text'].tolist(),
                'val': val['cleaned_text'].tolist(),
                'test': test['cleaned_text'].tolist()
            }, f)
        
        return output_path

    def process_twitter(self, zip_path: Path) -> Path:
        """Process Twitter dataset"""
        print(f"Processing Twitter data from {zip_path}")
        
        # Extract and process tweets
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        
        # Process each file in the extracted directory
        tweets = []
        for file in (RAW_DIR / "twitter_cikm_2010").glob("*.txt"):
            with open(file, 'r', encoding='utf-8') as f:
                tweets.extend(f.read().splitlines())
        
        # Clean and save
        cleaned_tweets = [self.clean_text(tweet) for tweet in tweets]
        output_path = PROCESSED_DIR / "twitter.json"
        with open(output_path, 'w') as f:
            json.dump(cleaned_tweets, f)
        
        return output_path

def main():
    """Main data preparation workflow"""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    
    preprocessor = DataPreprocessor()
    
    # Download datasets
    print("Downloading datasets...")
    twitter_url = "https://archive.org/download/twitter_cikm_2010/twitter_cikm_2010.zip"
    
    try:
        # Load MSMARCO from HuggingFace
        print("Loading MSMARCO from HuggingFace")
        try:
            # Load MSMARCO dataset from HuggingFace with version detection
            try:
                msmarco_dataset = load_dataset("microsoft/ms_marco", "v1.1")
                version = "v1.1"
            except:
                msmarco_dataset = load_dataset("microsoft/ms_marco", "v2.1") 
                version = "v2.1"
            
            # Extract relevant fields based on version
            docs = []
            for doc in tqdm(msmarco_dataset['train'], desc="Processing MSMARCO documents"):
                if version == "v1.1":
                    if 'passages' in doc and 'passage_text' in doc['passages'] and len(doc['passages']['passage_text']) > 0:
                        docs.append({
                            'docid': doc.get('docid', ''),
                            'url': doc.get('url', ''),
                            'title': doc.get('title', ''),
                            'text': doc['passages']['passage_text'][0]
                        })
                elif version == "v2.1":
                    if 'passage' in doc and len(doc['passage']) > 0:
                        docs.append({
                            'docid': doc.get('docid', ''),
                            'url': doc.get('url', ''),
                            'title': doc.get('title', ''),
                            'text': doc['passage']
                        })
            
            msmarco_df = pd.DataFrame(docs)
            print(f"Loaded {len(msmarco_df)} documents from MSMARCO dataset")
            
        except Exception as e:
            print(f"Error loading MSMARCO from HuggingFace: {str(e)}")
            print("Falling back to local file if available...")
            
            # Load MSMARCO from local file
            msmarco_path = RAW_DIR / "msmarco.tsv"
            if msmarco_path.exists():
                print(f"Loading MSMARCO from local file: {msmarco_path}")
                msmarco_df = pd.read_csv(msmarco_path, sep='\t', header=None, 
                                        names=['docid', 'url', 'title', 'text'])
            else:
                raise Exception("MSMARCO dataset not available locally. Please download it manually.")
        
        # Download Twitter data
        twitter_path = RAW_DIR / "twitter.zip" 
        if not twitter_path.exists():
            print(f"Downloading Twitter data from {twitter_url}")
            response = requests.get(twitter_url, stream=True)
            response.raise_for_status()
            with open(twitter_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192)):
                    f.write(chunk)
        
        print("Data download completed successfully")
        
        # Process datasets
        print("\nProcessing datasets...")
        try:
            msmarco_output = preprocessor.process_msmarco(msmarco_df)
            twitter_output = preprocessor.process_twitter(twitter_path)
            
            print(f"\nProcessing completed:")
            print(f"- MSMARCO data saved to {msmarco_output}")
            print(f"- Twitter data saved to {twitter_output}")
            
            # Generate combined dataset for SBERT fine-tuning
            with open(msmarco_output) as f:
                msmarco_data = json.load(f)
            with open(twitter_output) as f:
                twitter_data = json.load(f)
                
            combined_data = {
                'train': msmarco_data['train'] + twitter_data,
                'val': msmarco_data['val'],
                'test': msmarco_data['test']
            }
            
            combined_path = PROCESSED_DIR / "combined.json"
            with open(combined_path, 'w') as f:
                json.dump(combined_data, f)
                
            print(f"- Combined dataset saved to {combined_path}")
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return
        
        print("\nData preparation pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error downloading data: {str(e)}")

if __name__ == "__main__":
    main()
