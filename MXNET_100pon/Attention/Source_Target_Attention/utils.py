from sklearn.utils import shuffle


def create_corpus(inp, out):
    """
    input, ouputデータからコーパスをつくり、IDに置き換える関数

    Args
        inp: list
        out: list

    Return
        input_data: list
        output_data: list
        char2id: dict
        id2char: dict

    """

    # date.txtで登場するすべての文字にIDを割り当てる
    char2id = {c: i for i, c in
               enumerate(set(list("".join((inp + out)))))}

    id2char = {v: k for k, v in char2id.items()}

    input_data = []  # ID化された変換前日付データ
    output_data = []  # ID化された変換後日付データ
    for input_chars, output_chars in zip(inp, out):
        input_data.append([char2id[c] for c in input_chars])
        output_data.append([char2id[c] for c in output_chars])

    return input_data, output_data, char2id, id2char


def train2batch(input_data, output_data, batch_size=100):
    """
    データをバッチ化するための関数を定義
    """
    input_batch = []
    output_batch = []
    input_shuffle, output_shuffle = shuffle(input_data, output_data)
    for i in range(0, len(input_data), batch_size):
        input_batch.append(input_shuffle[i:i+batch_size])
        output_batch.append(output_shuffle[i:i+batch_size])
    return input_batch, output_batch


def date_load(f_path):
    input_date = []  # 変換前の日付データ
    output_date = []  # 変換後の日付データ

    # date.txtを1行ずつ読み込んで変換前と変換後に分割して、inputとoutputで分ける
    with open(f_path, "r") as f:
        date_list = f.readlines()
        for date in date_list:
            date = date[:-1]
            input_date.append(date.split("_")[0])
            output_date.append("_" + date.split("_")[1])

    return input_date, output_date
