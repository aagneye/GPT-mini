# tokenizer/tokenizer.py

import sentencepiece as spm
import os

class SPTokenizer:
    def __init__(
        self,
        model_file=None,
        data_path="data/dataset.txt",
        model_prefix="tokenizer/spm",
        vocab_size=16000,
    ):
        if model_file is not None:
            self.model_file = model_file
            train_prefix = os.path.splitext(self.model_file)[0]
        else:
            train_prefix = model_prefix
            self.model_file = model_prefix + ".model"

        if not os.path.exists(self.model_file):
            print("Training SentencePiece tokenizer...")
            spm.SentencePieceTrainer.train(
                input=data_path,
                model_prefix=train_prefix,
                vocab_size=vocab_size,
                character_coverage=0.9995,    # better for English
                model_type="bpe",             # IMPORTANT
                unk_id=0,
                pad_id=1,
                bos_id=2,
                eos_id=3,
            )

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_file)

        self.vocab_size = self.sp.get_piece_size()
        self.unk_id = self.sp.unk_id()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens):
        return self.sp.decode([int(t) for t in tokens])