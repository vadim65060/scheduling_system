def load_astg_jobs(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    n = int(lines[0])
    jobs = {}
    edges = {}

    idx = 1
    for _ in range(n):
        id_, t, r, q = map(int, lines[idx].split())
        jobs[id_] = (t, r, q)
        idx += 1

    for line in lines[idx:]:
        i, j, lij = map(int, line.split())
        edges[(i, j)] = lij

    return jobs, edges
