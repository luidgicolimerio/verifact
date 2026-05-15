"""
Verifica quais pacientes tiveram todos os note events inseridos no Qdrant.

Compara o CSV fonte com o que está na coleção ehr_noteevents e reporta:
- Pacientes com ingestão completa
- Pacientes com ingestão parcial (quantos faltam)
- Pacientes não inseridos
"""

import pandas as pd
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

QDRANT_URL = "http://localhost:6333"
COLLECTION = "ehr_noteevents"
CSV_PATH = "/home/lcolimerio/workspace/verifact/data/ehr_noteevents.csv.gz"


def get_qdrant_counts() -> dict[int, int]:
    """Retorna {SUBJECT_ID: count} de pontos no Qdrant."""
    client = QdrantClient(url=QDRANT_URL)
    counts = defaultdict(int)
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=None,
            limit=1000,
            offset=offset,
            with_payload=["SUBJECT_ID"],
            with_vectors=False,
        )
        for r in records:
            subject_id = r.payload.get("SUBJECT_ID")
            if subject_id is not None:
                counts[subject_id] += 1
        if offset is None:
            break

    return dict(counts)


def get_csv_counts() -> dict[int, int]:
    """Retorna {SUBJECT_ID: count} do CSV fonte."""
    df = pd.read_csv(CSV_PATH, usecols=["SUBJECT_ID"])
    return df["SUBJECT_ID"].value_counts().to_dict()


def main():
    print("Lendo CSV fonte...")
    csv_counts = get_csv_counts()

    print(f"Lendo Qdrant (coleção: {COLLECTION})...")
    qdrant_counts = get_qdrant_counts()

    print("\n=== Resultado da Verificação ===\n")

    complete, partial, missing = [], [], []

    for subject_id, expected in sorted(csv_counts.items()):
        inserted = qdrant_counts.get(subject_id, 0)
        if inserted == 0:
            missing.append((subject_id, expected))
        elif inserted < expected:
            partial.append((subject_id, inserted, expected, expected - inserted))
        else:
            complete.append(subject_id)

    print(f"✅ Completos : {len(complete)} pacientes")
    print(f"⚠️  Parciais  : {len(partial)} pacientes")
    print(f"❌ Não inseridos: {len(missing)} pacientes")

    if partial:
        print("\n--- Pacientes com ingestão PARCIAL ---")
        print(f"{'SUBJECT_ID':>12} {'Inseridos':>10} {'Esperados':>10} {'Faltando':>10}")
        for subject_id, inserted, expected, missing_ct in partial:
            print(f"{subject_id:>12} {inserted:>10} {expected:>10} {missing_ct:>10}")

    if missing:
        print("\n--- Pacientes NÃO inseridos ---")
        print(f"{'SUBJECT_ID':>12} {'Esperados':>10}")
        for subject_id, expected in missing:
            print(f"{subject_id:>12} {expected:>10}")


if __name__ == "__main__":
    main()
