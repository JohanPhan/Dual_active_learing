def normalizer(vector):
    vector = (vector - min(vector))/(max(vector)-min(vector))
    return vector
        