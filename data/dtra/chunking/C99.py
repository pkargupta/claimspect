from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import spacy
import numpy as np
import json

# load core english library
nlp = spacy.load("en_core_web_sm")

def cosine_sim(c1, c2):
    try:
        # works for Counter
        n1 = np.sqrt(sum([x * x for x in list(c1.values())]))
        n2 = np.sqrt(sum([x * x for x in list(c2.values())]))
        num = sum([c1[key] * c2[key] for key in c1])
    except:
        # works for ordinary list
        assert len(c1) == len(c2)
        n1 = np.sqrt(sum([x * x for x in c1]))
        n2 = np.sqrt(sum([x * x for x in c2]))
        num = sum([c1[i] * c2[i] for i in range(len(c1))])
    try:
        if n1 * n2 < 1e-9:  # divide by zero case
            return 0
        return num / (n1 * n2)
    except:
        return 0

class EnglishTokenizer:
    """
    A tokenizer is a class with tokenize(text) method
    """
    def __init__(self):
        pass

    def tokenize(self, text):
        return text.lower().split()

class C99:
    """
    Reference:
        "Advances in domain independent linear text segmentation"
    """
    def __init__(self, window, std_coeff, tokenizer=EnglishTokenizer()):
        self.window = window
        self.std_coeff = std_coeff
        self.tokenizer = tokenizer

    def segment(self, document, enable_tqdm=False):
        if len(document) < 3:
            return [1] + [0 for _ in range(len(document) - 1)]
        
        n = len(document)
        self.window = min(self.window, n)
        
        # Convert document embeddings into a NumPy array
        V = np.vstack(document)  # shape (n, d)
        
        # Compute norms of each vector
        norms = np.linalg.norm(V, axis=1)
        norms[norms == 0] = 1e-9  # Avoid division by zero
        
        # Compute the dot product matrix
        dot_products = V @ V.T  # shape (n, n)
        
        # Compute the outer product of norms
        norms_product = np.outer(norms, norms)  # shape (n, n)
        
        # Compute cosine similarities
        self.sim = np.divide(
            dot_products, norms_product, 
            out=np.zeros_like(dot_products), where=norms_product != 0
        )
        
        # Ensure the similarity matrix is symmetric
        self.sim = (self.sim + self.sim.T) / 2.0
        
        # Proceed with the rest of the method as before
        # Compute rank matrix & sum matrix
        self.rank = np.zeros((n, n))
        for i in tqdm(range(n), desc="Calculating Rank Matrix", disable=not enable_tqdm):
            for j in range(i, n):
                r1 = max(0, i - self.window + 1)
                r2 = min(n - 1, i + self.window - 1)
                c1 = max(0, j - self.window + 1)
                c2 = min(n - 1, j + self.window - 1)
                sublist = self.sim[r1:(r2 + 1), c1:(c2 + 1)].flatten()
                lowlist = np.sum(sublist < self.sim[i][j])
                self.rank[i][j] = 1.0 * lowlist / ((r2 - r1 + 1) * (c2 - c1 + 1))
                self.rank[j][i] = self.rank[i][j]

        # The rest of your method remains the same
        # ...

        # Calculate prefix sum matrix for fast range sum queries
        prefix_sm = np.zeros((n, n))
        for i in tqdm(range(n), desc="Calculating Prefix Sum Matrix", disable=not enable_tqdm):
            for j in range(n):
                prefix_sm[i][j] = self.rank[i][j]
                if i - 1 >= 0: prefix_sm[i][j] += prefix_sm[i - 1][j]
                if j - 1 >= 0: prefix_sm[i][j] += prefix_sm[i][j - 1]
                if i - 1 >= 0 and j - 1 >= 0: prefix_sm[i][j] -= prefix_sm[i - 1][j - 1]
        
        self.sm = np.zeros((n, n))
        for i in tqdm(range(n), desc="Calculating Sum Matrix", disable=not enable_tqdm):
            for j in range(i, n):
                if i == 0:
                    self.sm[i][j] = prefix_sm[j][j]
                else:
                    self.sm[i][j] = prefix_sm[j][j] - prefix_sm[i - 1][j] \
                                    - prefix_sm[j][i - 1] + prefix_sm[i - 1][i - 1]
                self.sm[j][i] = self.sm[i][j]

        # Determine boundaries
        D = 1.0 * self.sm[0][n - 1] / (n * n)
        darr, region_arr, idx = [D], [Region(0, n - 1, self.sm)], []
        sum_region, sum_area = float(self.sm[0][n - 1]), float(n * n)
        for i in tqdm(range(n - 1), desc="Determining Boundaries", disable=not enable_tqdm):
            mx, pos = -1e9, -1
            for j, region in enumerate(region_arr):
                if region.l == region.r:
                    continue
                region.split(self.sm)
                den = sum_area - region.area + region.lch.area + region.rch.area
                cur = (sum_region - region.tot + region.lch.tot + region.rch.tot) / den
                if cur > mx:
                    mx, pos = cur, j
            assert pos >= 0
            tmp = region_arr[pos]
            region_arr[pos] = tmp.rch
            region_arr.insert(pos, tmp.lch)
            sum_region += tmp.lch.tot + tmp.rch.tot - tmp.tot
            sum_area += tmp.lch.area + tmp.rch.area - tmp.area
            darr.append(sum_region / sum_area)
            idx.append(tmp.best_pos)

        dgrad = [(darr[i + 1] - darr[i]) for i in range(len(darr) - 1)]

        # Optional step, smooth gradient
        smooth_dgrad = [dgrad[i] for i in range(len(dgrad))]
        if len(dgrad) > 1:
            smooth_dgrad[0] = (dgrad[0] * 2 + dgrad[1]) / 3.0
            smooth_dgrad[-1] = (dgrad[-1] * 2 + dgrad[-2]) / 3.0
        for i in range(1, len(dgrad) - 1):
            smooth_dgrad[i] = (dgrad[i - 1] + 2 * dgrad[i] + dgrad[i + 1]) / 4.0
        dgrad = smooth_dgrad

        avg, stdev = np.average(dgrad), np.std(dgrad)
        cutoff = avg + self.std_coeff * stdev
        assert len(idx) == len(dgrad)
        above_cutoff_idx = [i for i in range(len(dgrad)) if dgrad[i] >= cutoff]
        boundary = idx[:max(above_cutoff_idx) + 1] if len(above_cutoff_idx) > 0 else []
        ret = [0 for _ in range(n)]
        for i in boundary:
            ret[i] = 1
            # Boundary should not be too close
            for j in range(i - 1, i + 2):
                if 0 <= j < n and j != i and ret[j] == 1:
                    ret[i] = 0
                    break
        return [1] + ret[:-1]

class Region:
    def __init__(self, l, r, sm_matrix):
        assert r >= l
        self.tot = sm_matrix[l][r]
        self.l = l
        self.r = r
        self.area = (r - l + 1) ** 2
        self.lch, self.rch, self.best_pos = None, None, -1

    def split(self, sm_matrix):
        if self.best_pos >= 0:
            return
        if self.l == self.r:
            self.best_pos = self.l
            return
        assert self.r > self.l
        mx, pos = -1e9, -1
        for i in range(self.l, self.r):
            carea = (i - self.l + 1) ** 2 + (self.r - i) ** 2
            cur = (sm_matrix[self.l][i] + sm_matrix[i + 1][self.r]) / carea
            if cur > mx:
                mx, pos = cur, i
        assert pos >= self.l and pos < self.r
        self.lch = Region(self.l, pos, sm_matrix)
        self.rch = Region(pos + 1, self.r, sm_matrix)
        self.best_pos = pos

class TopicSegmentor:
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        self.embedder = SentenceTransformer('allenai-specter')
        self.model = C99(window=3, std_coeff=0.6)

    def segment(self, context, enable_tqdm=False):
        pos = []
        sentences = []
        for c_id, sent in enumerate(context):
            doc = nlp(sent)
            sentence = []
            for s_id, s in enumerate(doc.sents):
                pos.append([c_id, s_id])
                sentence.append(str(s))
            sentences.append(sentence)

        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(len(sentences)), desc="Encoding Sentences", disable=not enable_tqdm):
                embedding = self.embedder.encode(sentences[i])
                embeddings.append(embedding)

        sent_label = []
        for i in tqdm(range(len(embeddings)), desc="Segmenting Topics", disable=not enable_tqdm):
            boundary = self.model.segment(embeddings[i], enable_tqdm)
            temp_labels = []
            l = 0
            for j in range(len(boundary)):
                if boundary[j] == 1:
                    l += 1
                temp_labels.append(l)
            sent_label.append(temp_labels)

        res = []
        for item in pos:
            context = sentences[item[0]]
            topic = sent_label[item[0]]
            assert len(context) == len(topic)
            exact_topic = topic[item[1]]

            subres = [context[t_id] for t_id in range(len(topic)) if topic[t_id] == exact_topic]
            if subres not in res:
                res.append(subres)
        
        return res

def main():
    
    corpus = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
    
    segmentor = TopicSegmentor()
    segments = segmentor.segment([corpus])

if __name__ == "__main__":
    main()