
def cross_sectional_preprocess(frame, feature_columns):
    out = frame.copy()
    for column in feature_columns:
        values = out.groupby("date")[column]
        median = values.transform("median"); mad = values.transform(lambda x: (x - x.median()).abs().median())
        clipped = out[column].clip(median - 5 * mad.replace(0, 1), median + 5 * mad.replace(0, 1))
        out[column] = (clipped - clipped.groupby(out["date"]).transform("mean")) / clipped.groupby(out["date"]).transform("std").replace(0, 1)
    return out
