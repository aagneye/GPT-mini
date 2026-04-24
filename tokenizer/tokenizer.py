# tokenizer/tokenizer.py

import sentencepiece as spm
import os

class SPTokenizer:
    def __init__(self, data_path="data/dataset.txt", model_prefix="tokenizer/spm", vocab_size=2000):
        
        self.model_file = model_prefix + ".model"

        # train tokenizer if not exists
        if not os.path.exists(self.model_file):
            print("Training SentencePiece tokenizer...")
            spm.SentencePieceTrainer.train(
                input=data_path,
                model_prefix=model_prefix,
                vocab_size=4000,              # increase vocab
                character_coverage=0.9995,    # better for English
                model_type="bpe",             # IMPORTANT
                unk_id=0,
                pad_id=1,
                bos_id=2,
                eos_id=3,
            )

        # load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_file)

        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, tokens):
        return self.sp.decode(tokens)