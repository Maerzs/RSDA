import json


def calculate_non_zero_mean(json_file_path):

    s_inp_depth_values = []
    s_inp_complex_values = []

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    for sample in data:
        if 'evaluation_scores' in sample:
            scores = sample['evaluation_scores']

            if 's_inp_depth' in scores and scores['s_inp_depth'] != 0:
                s_inp_depth_values.append(scores['s_inp_depth'])

            if 's_inp_complex' in scores and scores['s_inp_complex'] != 0:
                s_inp_complex_values.append(scores['s_inp_complex'])

    s_inp_depth_mean = sum(s_inp_depth_values) / len(s_inp_depth_values) if s_inp_depth_values else 0
    s_inp_complex_mean = sum(s_inp_complex_values) / len(s_inp_complex_values) if s_inp_complex_values else 0

    print(f"s_inp_depth no 0: {len(s_inp_depth_values)}")
    print(f"s_inp_depth no 0: {s_inp_depth_mean:.4f}")
    print(f"\ns_inp_complex no 0: {len(s_inp_complex_values)}")
    print(f"s_inp_complex no 0: {s_inp_complex_mean:.4f}")

    return {
        's_inp_depth': {
            'count': len(s_inp_depth_values),
            'mean': s_inp_depth_mean
        },
        's_inp_complex': {
            'count': len(s_inp_complex_values),
            'mean': s_inp_complex_mean
        }
    }

if __name__ == "__main__":
    result = calculate_non_zero_mean('/home/disk2/lyd/test/LLM/new/data/dolly/dolly_eval.json')