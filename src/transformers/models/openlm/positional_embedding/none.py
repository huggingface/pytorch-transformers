def identity_with_cast(q, k, v, offset: int = 0):
    return q.to(v.dtype), k.to(v.dtype), v