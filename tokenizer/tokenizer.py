# tokenizer/tokenizer.py

import sentencepiece as spm
import os

class SPTokenizer:
    def __init__(
        self,
        model_file=None,
        data_path="data/dataset.txt",
        model_prefix="tokenizer/spm",
        vocab_size=32000,
        byte_fallback=True,
        input_sentence_size=0,
    ):
        if model_file is not None:
            self.model_file = model_file
            train_prefix = os.path.splitext(self.model_file)[0]
        else:
            train_prefix = model_prefix
            self.model_file = model_prefix + ".model"

        if not os.path.exists(self.model_file):
            print("Training SentencePiece tokenizer...")
            train_kwargs = dict(
                input=data_path,
                model_prefix=train_prefix,
                vocab_size=vocab_size,
                character_coverage=0.9995,    # better for English
                model_type="bpe",             # IMPORTANT
                # byte_fallback lets any UTF-8 byte be represented, so web text
                # never hits <unk>. Critical for FineWeb-Edu at 32k vocab.
                byte_fallback=byte_fallback,
                unk_id=0,
                pad_id=1,
                bos_id=2,
                eos_id=3,
            )
            # On multi-GB corpora, subsample sentences for the trainer so it does
            # not try to load everything into RAM.
            if input_sentence_size:
                train_kwargs["input_sentence_size"] = input_sentence_size
                train_kwargs["shuffle_input_sentence"] = True
            spm.SentencePieceTrainer.train(**train_kwargs)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_file)

        self.vocab_size = self.sp.get_piece_size()
        self.unk_id = self.sp.unk_id()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens):
        return self.sp.decode([int(t) for t in tokens])

    def eos_id(self):
        return self.sp.eos_id()


def main():
    """CLI: train or load SPM using paths from config (repo root).

    To train a fresh 32k BPE tokenizer with byte_fallback on a FineWeb-Edu
    sample, point the env vars at the sample and delete any stale model:

        GPT_SPM_MODEL=tokenizer/spm32k.model \\
        GPT_TOKENIZER_TRAIN_INPUT=data/fineweb_sample.txt \\
        GPT_TOKENIZER_VOCAB_SIZE=32000 \\
        GPT_TOKENIZER_INPUT_SENTENCE_SIZE=2000000 \\
        python -m tokenizer
    """
    from config import data_path, spm_model_path

    train_input = os.environ.get("GPT_TOKENIZER_TRAIN_INPUT", data_path)
    vocab_size = int(os.environ.get("GPT_TOKENIZER_VOCAB_SIZE", "32000"))
    input_sentence_size = int(
        os.environ.get("GPT_TOKENIZER_INPUT_SENTENCE_SIZE", "0")
    )

    print("Tokenizer setup")
    print(f"  Train input: {train_input}")
    print(f"  SPM path: {spm_model_path}")
    print(f"  vocab_size (if training): {vocab_size}")
    tok = SPTokenizer(
        model_file=spm_model_path,
        data_path=train_input,
        vocab_size=vocab_size,
        byte_fallback=True,
        input_sentence_size=input_sentence_size,
    )
    print(f"  vocab_size: {tok.vocab_size}")
    print(f"  model_file: {tok.model_file}")
    print("Done.")


if __name__ == "__main__":
    main()