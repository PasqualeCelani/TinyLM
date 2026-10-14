import random


class GreedyDecoder:
    def __call__(self, distribution):
        if not distribution:
            return None
        return max(distribution, key=distribution.get)


class TemperatureSamplingDecoder:
    def __init__(self, temperature=1.0, top_k=None):
        self.temperature = temperature
        self.top_k = top_k

    def __call__(self, distribution):
        if not distribution:
            return None
        items = sorted(distribution.items(), key=lambda kv: -kv[1])
        if self.top_k:
            items = items[: self.top_k]
        scores = [p ** (1.0 / self.temperature) for _, p in items]
        return random.choices([tid for tid, _ in items], weights=scores, k=1)[0]


DECODERS = {
    "greedy": GreedyDecoder,
    "sample": TemperatureSamplingDecoder,
}


def build_decoder(config):
    name = config.get("decoding", "greedy")
    params = config.get("decoding_params", {})
    return DECODERS[name](**params)