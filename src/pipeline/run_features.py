from src.features.build_features import build_customer_feature_table

def main():
    out_path = build_customer_feature_table()
    print(out_path)

if __name__ == "__main__":
    main()
