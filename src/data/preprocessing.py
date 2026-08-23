import pickle
import pandas as pd
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"

MIN_USER_ORDERS = 4
MIN_PRODUCT_FREQ = 5


def load_raw_data(raw_dir=RAW_DIR):
    aisles = pd.read_csv(raw_dir / "aisles.csv")
    departments = pd.read_csv(raw_dir / "departments.csv")
    order_products_prior = pd.read_csv(raw_dir / "order_products_prior.csv")
    order_products_train = pd.read_csv(raw_dir / "order_products_train.csv")
    orders = pd.read_csv(raw_dir / "orders.csv")
    products = pd.read_csv(raw_dir / "products.csv")

    return aisles, departments, order_products_prior, order_products_train, orders, products


def build_sequence_table(order_products_prior, order_products_train, orders):
    full_data = pd.concat([order_products_prior, order_products_train])
    full_data.reset_index(drop=True, inplace=True)
    full_data = pd.merge(full_data, orders, on='order_id', how='left')

    cols = ['user_id', 'order_id', 'order_number', 'add_to_cart_order', 'product_id']

    return full_data[cols].copy()


def build_product_catalog(products, aisles, departments):
    catalog = products.merge(aisles, on='aisle_id')
    catalog = catalog.merge(departments, on='department_id')

    cols = ['product_id', 'product_name', 'aisle', 'department']

    return catalog[cols]


def filter_by_thresholds(sequence_data, min_user_orders=MIN_USER_ORDERS, min_product_freq=MIN_PRODUCT_FREQ):
    product_counts = sequence_data['product_id'].value_counts()
    valid_products = product_counts[product_counts >= min_product_freq].index
    sequence_data = sequence_data[sequence_data['product_id'].isin(valid_products)]

    user_order_counts = sequence_data.groupby('user_id')['order_number'].nunique()
    valid_users = user_order_counts[user_order_counts >= min_user_orders].index
    sequence_data = sequence_data[sequence_data['user_id'].isin(valid_users)]

    return sequence_data


def build_sequences(sequence_data):
    sequence_data = sequence_data.sort_values(['user_id', 'order_number', 'add_to_cart_order'])
    sequences = sequence_data.groupby('user_id')['product_id'].apply(list).to_dict()

    return sequences


def save_processed_data(sequences, catalog, output_dir=PROCESSED_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)

    catalog.to_csv(output_dir / "product_info.csv", index=False)


def main():
    aisles, departments, order_products_prior, order_products_train, orders, products = load_raw_data()
    catalog = build_product_catalog(products, aisles, departments)
    sequence_data = build_sequence_table(order_products_prior, order_products_train, orders)
    sequence_data = filter_by_thresholds(sequence_data)
    sequences = build_sequences(sequence_data)
    save_processed_data(sequences, catalog)


if __name__ == "__main__":
    main()
