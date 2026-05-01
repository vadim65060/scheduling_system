import os
import random
from jobs_loader import load_stg


def augment_from_stg(stg_path: str, lmax: int):
    """
    Загружает исходный STG и генерирует:
        r_i ∈ [1, 2000]
        q_i ∈ [1, 2000]
        L(i,j) ∈ [t_i, lmax]
    """

    jobs, precedence = load_stg(stg_path)

    # Генерируем r_i, q_i
    r = {}
    q = {}
    for job in jobs:
        t_i = int(job.d_i)
        r[job.id] = random.randint(1, 2000)
        q[job.id] = random.randint(1, 2000)

    # Генерируем L(i,j) на основе исходных дуг
    L = {}
    for (i, j), old_lij in precedence.items():
        t_i = int([job.d_i for job in jobs if job.id == i][0])
        L[(i, j)] = random.randint(t_i, max(t_i, lmax))

    return jobs, r, q, L


def save_augmented(path: str, jobs, r, q, L):
    """
    Сохраняет в формате:
        n
        id t_i r_i q_i
        i j L(i,j)
    """
    with open(path, "w") as f:
        f.write(f"{len(jobs)}\n")

        for job in sorted(jobs, key=lambda j: j.id):
            f.write(f"{job.id} {int(job.d_i)} {r[job.id]} {q[job.id]}\n")

        for (i, j), lij in sorted(L.items()):
            f.write(f"{i} {j} {lij}\n")


def process_folder(stg_folder: str, out_folder: str, lmax: int = 50):
    os.makedirs(out_folder, exist_ok=True)

    for fname in os.listdir(stg_folder):
        if not fname.endswith(".stg"):
            continue

        in_path = os.path.join(stg_folder, fname)
        out_path = os.path.join(out_folder, fname.replace(".stg", ".astg"))

        jobs, r, q, L = augment_from_stg(in_path, lmax)
        save_augmented(out_path, jobs, r, q, L)

        print(f"✓ {fname} → {out_path}")


if __name__ == "__main__":
    process_folder("data/100", "data/100_aug_l50", 50)
    process_folder("data/100", "data/100_aug_l75", 75)
    process_folder("data/100", "data/100_aug_l100", 100)
