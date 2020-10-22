from typing import List, Union, Tuple, Hashable, Dict, Any
from collections import defaultdict


def create_cluster(cluster: Dict[str, Any]) -> Dict[str, Union[List[Tuple], float, int]]:
    data = cluster['data'].copy()
    distances = sum([sum([d for d in label_distances]) / len(label_distances)
                     for label, label_distances in cluster['distances'].items()])
    return {'data': data, 'avg_distance': distances / len(cluster['distances']), 'min': data[0][1], 'max': data[-1][1]}

# def possible_clusters(sequences: List[Tuple[Hashable, Union[int, float]]], min_support: int,
#                       radius: float, add_edges: bool = True) \
#         -> List[Dict[str, Union[List[Tuple], float, int]]]:
#     clusters = []
#     sequences = sorted(sequences, key=lambda p: p[1])
#     for i, (label1, value1) in enumerate(sequences):
#         is_left_edge = abs(sequences[0][1] - value1) < radius and i > 0
#         is_right_edge = abs(sequences[-1][1] - value1) < radius
#         cluster = {'labels': set([]), 'data': [], 'distances': defaultdict(list)}
#         for label2, value2 in sequences[i:]:
#             if abs(value2 - value1) > radius:
#                 break
#             for label3, value3 in cluster['data']:
#                 cluster['distances'][label3].append(abs(value2 - value3))
#             cluster['labels'].add(label2)
#             cluster['data'].append((label2, value2))
#             if add_edges and len(cluster['labels']) >= min_support:
#                 if is_right_edge or i == 0:
#                     clusters.append(create_cluster(cluster))
#         if len(cluster['labels']) >= min_support and (cluster['data'][-1][1] > clusters[-1]['max'] or is_left_edge):
#             clusters.append(create_cluster(cluster))
#     clusters = sorted(clusters, key=lambda c: c['min'])
#     for i, cluster in zip(range(len(clusters)), clusters):
#         cluster.update({'id': i})
#     return clusters

def extract_clusters(sequences: List[Tuple[Hashable, Union[int, float]]], min_support: int, radius: float) \
        -> List[Dict[str, Union[List[Tuple], float, int]]]:
    clusters = []
    sequences = sorted(sequences, key=lambda p: p[1])
    for i, (label1, value1) in enumerate(sequences):
        cluster = {'labels': set([]), 'data': [], 'distances': defaultdict(list)}
        for label2, value2 in sequences[i:]:
            if abs(value2 - value1) > radius:
                break
            for label3, value3 in cluster['data']:
                cluster['distances'][label3].append(abs(value2 - value3))
            cluster['labels'].add(label2)
            cluster['data'].append((label2, value2))
            if len(cluster['labels']) >= min_support:
                clusters.append(create_cluster(cluster))
    clusters = sorted(clusters, key=lambda c: (c['min'], c['max']))
    for i, cluster in zip(range(len(clusters)), clusters):
        cluster.update({'id': i})
    return clusters


def find_candidates_for_each_cluster(clusters: List[Dict[str, Union[List[Tuple], float, int]]], radius: float):
    sorted_by_min = sorted(clusters, key=lambda c: (c['min'], c['max']))
    sorted_by_max = sorted(clusters, key=lambda c: (c['max'], c['min']))
    candidates = {c['id']: {'candidates': set([]), 'final': True, 'contained': set([])} for c in clusters}
    for cluster in sorted_by_min:
        for candidate in sorted_by_max:
            candidate_id = candidate['id']
            if candidate['max'] < cluster['min']: # also implies that candidate['min'] < cluster['min']
                candidates[cluster['id']]['candidates'] -= candidates[candidate_id]['candidates']
                if abs(cluster['max'] - candidate['min']) > radius:
                    # otherwise, there is a cluster containing cluster and candidate
                    candidates[cluster['id']]['candidates'].add(candidate_id)
                candidates[candidate_id]['final'] = False
            elif cluster['min'] < candidate['min']:
                break
            elif candidate['min'] <= cluster['min'] and cluster['max'] <= candidate['max']:
                candidates[cluster['id']]['contained'].add(candidate_id)

    return candidates


def find_best_candidate(cached):
    argmin_ids, min_sum_coverage = [], None
    for candidate_argmin_ids, candidate_min_sum_coverage in cached:
        if min_sum_coverage is None:
            argmin_ids, min_sum_coverage = candidate_argmin_ids, candidate_min_sum_coverage
            continue
        elif min_sum_coverage >= candidate_min_sum_coverage:
            argmin_ids, min_sum_coverage = candidate_argmin_ids, candidate_min_sum_coverage
    return argmin_ids, min_sum_coverage


def minimum_avg_distances_coverage(sequences: List[Tuple[Hashable, Union[int, float]]], min_support: int,
                                   radius: float) \
        -> List[List[Tuple[Hashable, Union[int, float]]]]:
    clusters = extract_clusters(sequences, min_support, radius)
    candidates = find_candidates_for_each_cluster(clusters, radius)
    cached = {}
    min_value = clusters[0]['min']
    for cluster in clusters:
        cluster_id = cluster['id']
        cluster_candidates = candidates[cluster_id]['candidates']
        if abs(cluster['min'] - min_value) >= radius and len(cluster_candidates) == 0:
            continue
        filtered_candidates = cluster_candidates.copy()
        for candidate_id in cluster_candidates:
            if candidate_id not in cached:
                filtered_candidates.remove(candidate_id)
            elif len(candidates[candidate_id]['contained']) > 0:
                candidate_max_cached = max([clusters[cid]['max'] for cid in cached[candidate_id][0]])
                for contains_id in candidates[candidate_id]['contained']:
                    if clusters[contains_id]['min'] == clusters[candidate_id]['min']:
                        filtered_candidates.remove(candidate_id)
                        break
                    elif candidate_max_cached <= clusters[contains_id]['min']:
                        filtered_candidates.remove(candidate_id)
                        break
        argmin_ids, min_sum_coverage = find_best_candidate([cached[i] for i in filtered_candidates])
        if min_sum_coverage is None:
            cached[cluster_id] = ([cluster_id], cluster['avg_distance'])
        else:
            cached[cluster_id] = (argmin_ids + [cluster_id], min_sum_coverage + cluster['avg_distance'])
    argmin_ids, min_sum_coverage = find_best_candidate([cached[i] for i, c in candidates.items()
                                                        if c['final'] and i in cached])
    return [cluster['data'] for cluster in clusters if cluster['id'] in argmin_ids]


def align_sequences(sequences: List[Tuple[Hashable, Union[int, float]]], min_support: int,
                                   radius: float, duplicates: str = 'min'):
    labels = set([label for label, _ in sequences])
    clusters = minimum_avg_distances_coverage(sequences, min_support, radius)
    aligned_sequences = []
    for cluster in clusters:
        event = {l: [] for l in labels}
        for l, v in cluster:
            event[l].append(v)
        for l, vs in event.items():
            if len(vs) == 0:
                event[l] = None
            elif duplicates == 'min':
                event[l] = min(vs)
            elif duplicates == 'max':
                event[l] = max(vs)
            elif duplicates == 'mean':
                event[l] = sum(vs) / len(vs)
            elif event[l] == 'median':
                event[l] = sorted(vs)[int(len(vs)//2)]
        aligned_sequences.append(event)

# population = [('A', -2), ('B', 0), ('C', 2), ('A', 2), ('B', 2), ('B', 3), ('A', 4), ('C', 6)]
# minimum_avg_distances_coverage(population, 2, 3)