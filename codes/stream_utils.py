from collections import Counter
import logging
import numpy as np
import os

from utils import read_triple_raw


class KBStream:
    def __init__(self, data_path, test_file_name=None, stream_init_proportion=0.5, n_stream_updates=10,
                 frac_old_train_samples=0.1, sample_nbh=False, seed=42):
        self.data_path = data_path
        self.stream_init_proportion = stream_init_proportion
        self.n_stream_updates = n_stream_updates
        self.frac_old_train_samples = frac_old_train_samples
        self.sample_nbh = sample_nbh
        self.stream_rng = np.random.default_rng(seed)
        self.train_rng = np.random.default_rng(seed)

        self.entity_set, self.relation_set = set(), set()

        with open(os.path.join(self.data_path, 'entities.dict')) as fin:
            for line in fin:
                eid, entity = line.strip().split('\t')
                self.entity_set.add(entity)

        with open(os.path.join(self.data_path, 'relations.dict')) as fin:
            for line in fin:
                rid, relation = line.strip().split('\t')
                self.relation_set.add(relation)

        if test_file_name is None or test_file_name == '':
            test_file_name = 'test.txt'

        self.train_triples = read_triple_raw(os.path.join(data_path, 'train.txt'))
        self.valid_triples = read_triple_raw(os.path.join(data_path, 'valid.txt'))
        self.test_triples = read_triple_raw(os.path.join(data_path, test_file_name))
        self.kb_state = {'entity2id': {}, 'relation2id': {},
                         'train_triples': [], 'valid_triples': [], 'test_triples': []}

    def get_init_kb(self):
        # INIT
        # Sample 10% of the most common nodes (hubs)
        # Sample (stream_init_proportion - 10)% of the remaining nodes randomly
        node_usage_train = Counter([e for (e, _, _) in self.train_triples] + [e for (_, _, e) in self.train_triples])
        init_entities = [_ent for _ent, _ in node_usage_train.most_common(len(node_usage_train) // 10)]
        for _ent in init_entities:
            del node_usage_train[_ent]
        permutation = self.stream_rng.permutation(len(node_usage_train))
        usage_list = list(node_usage_train.most_common())
        sample_size = int(np.ceil(max(self.stream_init_proportion - 0.1, 0.0)*len(self.entity_set)))
        init_entities.extend([usage_list[j][0] for j in permutation[:sample_size]])
        assert len(init_entities) == len(set(init_entities))
        init_entities = set(init_entities)

        entity2id, relation2id = {}, {}
        for eid, entity in enumerate(init_entities):
            entity2id[entity] = eid

        edge_coverage = {'train': 0, 'valid': 0, 'test': 0}
        init_train_triples, init_valid_triples, init_test_triples = [], [], []
        for edge in self.train_triples:
            e1, r, e2 = edge
            if e1 in init_entities and e2 in init_entities:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                init_train_triples.append((entity2id[e1], relation2id[r], entity2id[e2]))
                edge_coverage['train'] += 1

        for edge in self.valid_triples:
            e1, r, e2 = edge
            if e1 in init_entities and e2 in init_entities:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                init_valid_triples.append((entity2id[e1], relation2id[r], entity2id[e2]))
                edge_coverage['valid'] += 1

        for edge in self.test_triples:
            e1, r, e2 = edge
            if e1 in init_entities and e2 in init_entities:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                init_test_triples.append((entity2id[e1], relation2id[r], entity2id[e2]))
                edge_coverage['test'] += 1

        print(f"[STREAM] Init edge_coverage: "
              f"train: {edge_coverage['train']} ({edge_coverage['train'] / len(self.train_triples) * 100:0.2f}%) "
              f"valid: {edge_coverage['valid']} ({edge_coverage['valid'] / len(self.valid_triples) * 100:0.2f}%) "
              f"test: {edge_coverage['test']} ({edge_coverage['test'] / len(self.test_triples) * 100:0.2f}%)")
        print(f'[STREAM] Init entity_coverage:'
              f' {len(init_entities)} ({len(init_entities) / (len(self.entity_set)) * 100:0.2f}%)')

        self.kb_state['entity2id'] = entity2id.copy()
        self.kb_state['relation2id'] = relation2id.copy()
        self.kb_state['train_triples'] = init_train_triples.copy()
        self.kb_state['valid_triples'] = init_valid_triples.copy()
        self.kb_state['test_triples'] = init_test_triples.copy()

        return entity2id, relation2id, init_train_triples + init_valid_triples + init_test_triples, init_train_triples,\
               init_valid_triples, init_test_triples

    def batch_generator(self):
        for step in range(self.n_stream_updates):
            print(f'[STREAM] Generating batch {step + 1}...')
            entity2id, relation2id = self.kb_state['entity2id'], self.kb_state['relation2id']
            curr_train_triples, curr_valid_triples, curr_test_triples = \
                self.kb_state['train_triples'], self.kb_state['valid_triples'], self.kb_state['test_triples']
            new_train_triples, new_valid_triples, new_test_triples = [], [], []

            nentity_old = len(entity2id)
            seen_entities = set(entity2id.keys())
            unseen_entities = sorted(self.entity_set.difference(seen_entities))
            permutation = self.stream_rng.permutation(len(unseen_entities))
            sample_size = int(np.ceil((1 - self.stream_init_proportion) / self.n_stream_updates * len(self.entity_set)))
            if step == self.n_stream_updates - 1:
                sample_size = len(unseen_entities)
            new_entities = [unseen_entities[j] for j in permutation[:sample_size]]
            new_entities = set(new_entities)
            nentity_new = len(new_entities)

            for entity in new_entities:
                if entity not in entity2id:
                    entity2id[entity] = len(entity2id)
            assert (len(entity2id) - nentity_old) == nentity_new

            for edge in self.train_triples:
                e1, r, e2 = edge
                if e1 in seen_entities and e2 in seen_entities:
                    continue
                if (e1 in new_entities or e1 in seen_entities) and (e2 in new_entities or e2 in seen_entities):
                    if r not in relation2id:
                        relation2id[r] = len(relation2id)
                    new_train_triples.append((entity2id[e1], relation2id[r], entity2id[e2]))

            for edge in self.valid_triples:
                e1, r, e2 = edge
                if e1 in seen_entities and e2 in seen_entities:
                    continue
                if (e1 in new_entities or e1 in seen_entities) and (e2 in new_entities or e2 in seen_entities):
                    if r not in relation2id:
                        relation2id[r] = len(relation2id)
                    new_valid_triples.append((entity2id[e1], relation2id[r], entity2id[e2]))

            for edge in self.test_triples:
                e1, r, e2 = edge
                if e1 in seen_entities and e2 in seen_entities:
                    continue
                if (e1 in new_entities or e1 in seen_entities) and (e2 in new_entities or e2 in seen_entities):
                    if r not in relation2id:
                        relation2id[r] = len(relation2id)
                    new_test_triples.append((entity2id[e1], relation2id[r], entity2id[e2]))

            if self.frac_old_train_samples > 0:
                if self.sample_nbh:
                    sample_weights = np.ones(len(curr_train_triples))
                    nbh_entities = set([e1 for e1, _, _ in new_train_triples] + [e2 for _, _, e2 in new_train_triples])
                    nbh_entities.difference_update(unseen_entities)
                    for _idx, (e1, r, e2) in enumerate(curr_train_triples):
                        if e1 in nbh_entities or e2 in nbh_entities:
                            sample_weights[_idx] = 10.
                    sample_weights /= np.sum(sample_weights)
                else:
                    sample_weights = np.ones(len(curr_train_triples))
                    sample_weights /= np.sum(sample_weights)
                chosen_idx = self.train_rng.choice(np.arange(len(curr_train_triples)),
                                                   int(len(curr_train_triples) * self.frac_old_train_samples),
                                                   replace=False, p=sample_weights)
                sampled_train_triples = [curr_train_triples[_idx] for _idx in chosen_idx]
            else:
                sampled_train_triples = []
            sampled_train_triples += new_train_triples

            all_train_triples = new_train_triples + curr_train_triples
            all_valid_triples = new_valid_triples + curr_valid_triples
            all_test_triples = new_test_triples + curr_test_triples
            print(f"[STREAM] Batch edge_coverage: "
                  f"train: {len(new_train_triples)} ({len(new_train_triples) / len(self.train_triples) * 100:0.2f}%) "
                  f"valid: {len(new_valid_triples)} ({len(new_valid_triples) / len(self.valid_triples) * 100:0.2f}%) "
                  f"test: {len(new_test_triples)} ({len(new_test_triples) / len(self.test_triples) * 100:0.2f}%)")
            print(f"[STREAM] Sampled train size: "
                  f"{len(sampled_train_triples)} ({len(sampled_train_triples) / len(self.train_triples) * 100:0.2f}%)")
            print(f"[STREAM] Total edge_coverage: "
                  f"train: {len(all_train_triples)} ({len(all_train_triples) / len(self.train_triples) * 100:0.2f}%) "
                  f"valid: {len(all_valid_triples)} ({len(all_valid_triples) / len(self.valid_triples) * 100:0.2f}%) "
                  f"test: {len(all_test_triples)} ({len(all_test_triples) / len(self.test_triples) * 100:0.2f}%)")
            print(f'[STREAM] Total entity_coverage:'
                  f' {len(entity2id)} ({len(entity2id) / (len(self.entity_set)) * 100:0.2f}%)')

            self.kb_state['entity2id'] = entity2id.copy()
            self.kb_state['relation2id'] = relation2id.copy()
            self.kb_state['train_triples'] = all_train_triples.copy()
            self.kb_state['valid_triples'] = all_valid_triples.copy()
            self.kb_state['test_triples'] = all_test_triples.copy()

            yield entity2id, relation2id, all_train_triples + all_valid_triples + all_test_triples, \
                  sampled_train_triples, all_valid_triples, new_valid_triples, all_test_triples, new_test_triples, \
                  nentity_old, nentity_new
