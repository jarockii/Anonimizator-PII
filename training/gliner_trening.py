

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import ner_masker  # noqa: E402

DANE = ROOT / "training" / "dane"
LORA = ROOT / "modele" / "gliner-pl-lora"     # robocze nakładki z treningu
WYNIK = ROOT / "modele" / "gliner-pl"         # gotowy, scalony model


def main():
    parser = argparse.ArgumentParser(description="Trening LoRA GLiNER-a po polsku")
    parser.add_argument("--epoki", type=int, default=3, help="ile razy model przejdzie przez dane")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate – wielkość kroku uczenia")
    parser.add_argument("--batch", type=int, default=8, help="ile przykładów naraz")
    parser.add_argument("--wyjscie", type=Path, default=WYNIK,
                        help="katalog gotowego modelu, np. modele/gliner-pl-v2 (nakładki obok: <katalog>-lora)")
    args = parser.parse_args()
    wynik_dir = args.wyjscie if args.wyjscie.is_absolute() else ROOT / args.wyjscie
    # nakładki LoRA obok modelu – kolejny trening nie nadpisze poprzedniego
    lora_dir = LORA if wynik_dir == WYNIK else wynik_dir.with_name(wynik_dir.name + "-lora")

    for plik in ("gliner_train.jsonl", "gliner_val.jsonl"):
        if not (DANE / plik).exists():
            sys.exit(f"Brak {DANE / plik}. Najpierw uruchom: python training/gliner_dane.py")

    # Importy ciężkich bibliotek dopiero tutaj – błąd w argumentach wyjdzie od razu.
    from gliner2 import GLiNER2
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
    from peft import PeftModel

    print(f"1/3 Wczytuję oryginalny model: {ner_masker.NER_MODEL}")
    model = GLiNER2.from_pretrained(ner_masker.NER_MODEL)

    config = TrainingConfig(
        output_dir=str(lora_dir),
        num_epochs=args.epoki,
        batch_size=args.batch,
        eval_batch_size=16,
        task_lr=args.lr,               # przy LoRA to jedyny learning rate, który działa
        use_lora=True,
        lora_r=16,                     # „szerokość” nakładki
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["encoder"],   # uczymy tylko część czytającą tekst
        save_adapter_only=True,        # zapisujemy samą nakładkę (kilka MB)
        eval_strategy="epoch",         # sprawdzaj walidację po każdej epoce
        save_best=True,                # zachowaj najlepszą epokę (najniższy eval_loss)
        fp16=False, bf16=False,        # na CPU tylko pełna precyzja
        logging_steps=25,
        seed=42,
    )

    print(f"2/3 Trening: {args.epoki} epoki, batch {args.batch}, lr {args.lr} (CPU)…")
    start = time.time()
    trener = ExtractorTrainer(model, config)
    wynik = trener.train(train_data=str(DANE / "gliner_train.jsonl"),
                         eval_data=str(DANE / "gliner_val.jsonl"))
    minuty = (time.time() - start) / 60
    print(f"   gotowe w {minuty:.1f} min, najlepszy eval_loss: {wynik.get('best_metric')}")

    najlepsza = lora_dir / "best"
    if not najlepsza.exists():
        najlepsza = lora_dir / "final"
    print(f"3/3 Scalam nakładkę {najlepsza.relative_to(ROOT)} z modelem i zapisuję…")
    baza = GLiNER2.from_pretrained(ner_masker.NER_MODEL)
    scalony = PeftModel.from_pretrained(baza, str(najlepsza)).merge_and_unload()
    scalony.save_pretrained(str(wynik_dir))
    print(f"\nModel zapisany w {wynik_dir.relative_to(ROOT)}")
    print("Pomiar PO treningu:")
    print(f"  ANONIMIZATOR_NER_MODEL={wynik_dir.relative_to(ROOT)} python eval/szara_strefa.py --nazwa po")


if __name__ == "__main__":
    main()
