# simple_nsga2_variable_packs.py
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# ---------- problem data ----------
N_CELLS = 11000
CELLS_PER_PACK = 100
MAX_PACKS = N_CELLS // CELLS_PER_PACK   # 可变 pack 数上限 (110/10 = 11)
UNUSED_LABEL = MAX_PACKS
TARGET_CAPACITY = 3.0 * CELLS_PER_PACK  # 3Ah per cell * 10 = 30.0

np.random.seed(42)
random.seed(42)

# synthetic cell features: capacity (Ah), dcr (mOhm), sdr (%)
cells = np.zeros((N_CELLS, 3))
cells[:, 0] = np.random.normal(3.1, 0.1, N_CELLS)   # capacity
cells[:, 1] = np.random.normal(1.5, 0.05, N_CELLS)  # dcr
cells[:, 2] = np.random.normal(0.02, 0.005, N_CELLS) # sdr

EMPTY_PENALTY = 1e3

# ---------- evaluate (with hard-capacity constraint handled as G in pymoo style) ----------
def evaluate_individual(ind):
    """
    返回目标 (f1, f2)：
      f1 = sum of per-pack variances (cap,dcr,sdr) for non-empty packs (越小越好)
      f2 = - num_packs_used (越小越好 -> 更大 num_packs_used)
    同时可计算约束向量 G：对于每非空 pack 添加约束 TARGET_CAPACITY - cap_sum <= 0
    """
    packs = [[] for _ in range(MAX_PACKS)]
    for i, lab in enumerate(ind):
        if lab != UNUSED_LABEL and 0 <= lab < MAX_PACKS:
            packs[int(lab)].append(i)
    f1 = 0.0
    num_packs_used = 0
    for p in packs:
        if len(p) == 0:
            continue
        num_packs_used += 1
        cap = cells[p, 0]
        dcr = cells[p, 1]
        sdr = cells[p, 2]
        f1 += float(np.var(cap)) + float(np.var(dcr)) + float(np.var(sdr))
    f2 = - float(num_packs_used)
    return (f1, f2)

def constraint_violations(ind):
    """
    返回列表 G，G[i] <= 0 表示满足约束。
    对每个 pack（非空）有两个约束：
      1) count must equal CELLS_PER_PACK -> represent via two inequalities (cnt - target <=0 and target - cnt <=0)
         but since we enforce repair to exact counts, these usually zero. For empty pack we set zeros.
      2) capacity constraint: TARGET_CAPACITY - cap_sum <= 0  (hard constraint)
    我们返回长度 3*MAX_PACKS 的向量： [cnt_minus, target_minus_cnt, capacity_deficit] per pack
    """
    packs = [[] for _ in range(MAX_PACKS)]
    for i, lab in enumerate(ind):
        if lab != UNUSED_LABEL and 0 <= lab < MAX_PACKS:
            packs[int(lab)].append(i)
    G = []
    for p in packs:
        cnt = len(p)
        if cnt == 0:
            # no constraints for empty pack
            G.append(0.0)
            G.append(0.0)
            G.append(0.0)
        else:
            cap_sum = float(cells[p,0].sum())
            G.append(float(cnt - CELLS_PER_PACK))       # <=0
            G.append(float(CELLS_PER_PACK - cnt))       # <=0
            G.append(float(TARGET_CAPACITY - cap_sum))  # <=0 means cap_sum >= TARGET_CAPACITY
    return np.array(G, dtype=float)

# ---------- helpers: non-dominated sort & crowding distance (same as before) ----------
def nondominated_sort(pop_objs):
    S = [[] for _ in range(len(pop_objs))]
    n = [0]*len(pop_objs)
    fronts = [[]]
    for p in range(len(pop_objs)):
        for q in range(len(pop_objs)):
            if p == q: continue
            # p dominates q?
            less_equal = all(a <= b for a,b in zip(pop_objs[p], pop_objs[q]))
            strictly_less = any(a < b for a,b in zip(pop_objs[p], pop_objs[q]))
            if less_equal and strictly_less:
                S[p].append(q)
            else:
                le2 = all(a >= b for a,b in zip(pop_objs[p], pop_objs[q]))
                st2 = any(a > b for a,b in zip(pop_objs[p], pop_objs[q]))
                if le2 and st2:
                    n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if len(fronts[-1]) == 0:
        fronts.pop()
    return fronts

def crowding_distance(front, pop_objs):
    l = len(front)
    distances = {idx:0.0 for idx in front}
    if l == 0:
        return distances
    num_obj = len(pop_objs[0])
    for m in range(num_obj):
        vals = [(idx, pop_objs[idx][m]) for idx in front]
        vals.sort(key=lambda x: x[1])
        distances[vals[0][0]] = float('inf')
        distances[vals[-1][0]] = float('inf')
        vmin = vals[0][1]; vmax = vals[-1][1]
        if vmax == vmin: continue
        for i in range(1, l-1):
            prev = vals[i-1][1]; nex = vals[i+1][1]
            distances[vals[i][0]] += (nex - prev) / (vmax - vmin)
    return distances

# ---------- repair & operators (ensure non-empty packs have exactly CELLS_PER_PACK) ----------
def repair_individual(ind):
    labels = list(ind)
    # collect indices by label
    packs_idx = {p:[] for p in range(MAX_PACKS)}
    unused_idx = []
    for i, lab in enumerate(labels):
        if lab != UNUSED_LABEL and 0 <= lab < MAX_PACKS:
            packs_idx[int(lab)].append(i)
        else:
            unused_idx.append(i)
    # remove extras -> move to unused
    for p in range(MAX_PACKS):
        while len(packs_idx[p]) > CELLS_PER_PACK:
            idx = packs_idx[p].pop()
            labels[idx] = UNUSED_LABEL
            unused_idx.append(idx)
    # fill shortages by using unused indices
    for p in range(MAX_PACKS):
        while len(packs_idx[p]) < CELLS_PER_PACK:
            if not unused_idx:
                # no unused left; steal from a pack with surplus (shouldn't happen due to above)
                found = False
                for q in range(MAX_PACKS):
                    if len(packs_idx[q]) > CELLS_PER_PACK:
                        idx = packs_idx[q].pop()
                        labels[idx] = p
                        packs_idx[p].append(idx)
                        found = True
                        break
                if not found:
                    # forced assignment from random
                    idx = random.randrange(N_CELLS)
                    prev = labels[idx]
                    if prev != UNUSED_LABEL and 0 <= prev < MAX_PACKS:
                        if idx in packs_idx[prev]:
                            packs_idx[prev].remove(idx)
                    labels[idx] = p
                    packs_idx[p].append(idx)
            else:
                idx = unused_idx.pop()
                labels[idx] = p
                packs_idx[p].append(idx)
    return np.array(labels, dtype=int)

def uniform_crossover(a, b):
    child = a.copy()
    mask = np.random.rand(len(a)) < 0.5
    child[mask] = b[mask]
    child = repair_individual(child)
    return child

def mutate(ind, mut_rate=0.02):
    ind = ind.copy()
    for i in range(len(ind)):
        if random.random() < mut_rate:
            new_label = random.randint(0, MAX_PACKS)  # includes UNUSED_LABEL
            ind[i] = new_label
    ind = repair_individual(ind)
    return ind

# initial individual: choose random k packs (1..MAX_PACKS) and fill them
def random_feasible_individual():
    k = random.randint(1, MAX_PACKS)  # at least 1 pack
    labels = [UNUSED_LABEL]*N_CELLS
    all_indices = list(range(N_CELLS))
    random.shuffle(all_indices)
    ptr = 0
    for p in range(k):
        for _ in range(CELLS_PER_PACK):
            idx = all_indices[ptr]
            labels[idx] = p
            ptr += 1
    return np.array(labels, dtype=int)

# ---------- NSGA-II main loop (lightweight) ----------
def simple_nsga2(pop_size=100, generations=100, mut_rate=0.02):
    pop = [random_feasible_individual() for _ in range(pop_size)]
    pop_objs = [evaluate_individual(ind) for ind in pop]
    pop_G = [constraint_violations(ind) for ind in pop]

    for gen in range(1, generations+1):
        offspring = []
        for _ in range(pop_size):
            p1, p2 = random.sample(range(pop_size), 2)
            child = uniform_crossover(pop[p1], pop[p2])
            child = mutate(child, mut_rate=mut_rate)
            offspring.append(child)
        off_objs = [evaluate_individual(ind) for ind in offspring]
        off_G = [constraint_violations(ind) for ind in offspring]

        combined = pop + offspring
        combined_objs = pop_objs + off_objs
        combined_G = pop_G + off_G

        # filter infeasible first: keep those with all G <= 0 (feasible)
        feasible_idx = [i for i,g in enumerate(combined_G) if np.all(g <= 0)]
        infeasible_idx = [i for i,g in enumerate(combined_G) if not np.all(g <= 0)]

        # we prefer feasible individuals: build fronts among feasible first
        new_pop = []
        new_objs = []
        # if there are feasible solutions, do nondominated sort among feasible only
        if feasible_idx:
            feas_objs = [combined_objs[i] for i in feasible_idx]
            fronts = nondominated_sort(feas_objs)
            # map indices back
            selected = []
            for f in fronts:
                if len(new_pop) + len(f) > pop_size:
                    dist = crowding_distance(f, feas_objs)
                    f_sorted = sorted(f, key=lambda idx: (dist[idx] if dist[idx] != float('inf') else 1e9), reverse=True)
                    needed = pop_size - len(new_pop)
                    for idx in f_sorted[:needed]:
                        sel_idx = feasible_idx[idx]
                        new_pop.append(combined[sel_idx])
                        new_objs.append(combined_objs[sel_idx])
                    break
                else:
                    for idx in f:
                        sel_idx = feasible_idx[idx]
                        new_pop.append(combined[sel_idx])
                        new_objs.append(combined_objs[sel_idx])
        # if still slots left, fill from infeasible by least total violation then by nondom
        if len(new_pop) < pop_size and infeasible_idx:
            # sort infeasible by sum of positive violations (smaller better)
            viol_sums = [(i, np.sum(np.maximum(combined_G[i], 0.0))) for i in infeasible_idx]
            viol_sums.sort(key=lambda x: x[1])
            for idx,_ in viol_sums:
                if len(new_pop) >= pop_size:
                    break
                new_pop.append(combined[idx])
                new_objs.append(combined_objs[idx])

        # ensure we have pop_size
        if len(new_pop) < pop_size:
            # fill randomly from combined
            for i in range(len(combined)):
                if len(new_pop) >= pop_size:
                    break
                if combined[i] not in new_pop:
                    new_pop.append(combined[i])
                    new_objs.append(combined_objs[i])

        pop = new_pop
        pop_objs = new_objs
        pop_G = [constraint_violations(ind) for ind in pop]

        if gen % max(1, generations//10) == 0 or gen <= 5:
            best = min(pop_objs, key=lambda x: (x[0], x[1]))
            feas_count = sum(1 for g in pop_G if np.all(g <= 0))
            print(f"gen {gen:3d} | pop_size {len(pop)} | best f (f1,f2) = ({best[0]:.4f}, {best[1]:.4f}) | feasible {feas_count}/{len(pop)}")

    # final Pareto among feasible if exist, else among population
    feasible_idx = [i for i,g in enumerate(pop_G) if np.all(g <= 0)]
    if feasible_idx:
        feas_objs = [pop_objs[i] for i in feasible_idx]
        fronts = nondominated_sort(feas_objs)
        pareto_idx_local = fronts[0]
        pareto_idx = [feasible_idx[i] for i in pareto_idx_local]
    else:
        fronts = nondominated_sort(pop_objs)
        pareto_idx = fronts[0]

    pareto_sols = [pop[i] for i in pareto_idx]
    pareto_objs = [pop_objs[i] for i in pareto_idx]
    return pareto_sols, pareto_objs

# ---------- run and print results ----------
if __name__ == "__main__":
    pop_size = 120
    generations = 300
    pareto_sols, pareto_objs = simple_nsga2(pop_size=pop_size, generations=generations, mut_rate=0.03)

    print(f"\nFound Pareto size = {len(pareto_sols)}")
    # print details for first few Pareto solutions
    PRINT_PARETO = 10
    for idx, (sol, obj) in enumerate(sorted(zip(pareto_sols, pareto_objs), key=lambda x: (x[1][0], x[1][1]))):
        if PRINT_PARETO is not None and idx >= PRINT_PARETO:
            break
        print(f"\n--- Pareto solution #{idx}  (f1={obj[0]:.4f}, f2={obj[1]:.4f}) ---")
        packs = [[] for _ in range(MAX_PACKS)]
        unused = []
        for i, lab in enumerate(sol):
            if lab != UNUSED_LABEL and 0 <= lab < MAX_PACKS:
                packs[int(lab)].append(i)
            else:
                unused.append(i)
        for p in range(MAX_PACKS):
            members = packs[p]
            if len(members) == 0:
                continue
            caps = cells[members, 0]
            cap_sum = caps.sum()
            cap_mean = caps.mean()
            cap_var = caps.var()
            print(f" Pack {p}: count={len(members)}, cap_sum={cap_sum:.3f}, mean={cap_mean:.3f}, var={cap_var:.6f}")
            print("  indices:", members)
        print(" Unused count:", len(unused), " indices:", unused)

    # save 3D pareto plot (f1, -f2 as positive pack count)
    try:
        arr = np.array(pareto_objs)
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(arr[:,0], -arr[:,1], c='b', marker='o')  # -f2 -> pack count positive
        ax.set_xlabel('f1: sum variances')
        ax.set_ylabel('pack_count (positively shown)')
        plt.title('Pareto front (variance vs pack count)')
        plt.tight_layout()
        plt.savefig("pareto_variable_packs.png", dpi=200)
        print("Saved pareto_variable_packs.png")
    except Exception as e:
        print("Plotting failed:", e)
