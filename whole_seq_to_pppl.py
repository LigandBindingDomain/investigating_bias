"""
how to run:

For a single sequence:
python3 whole_seq_to_pppl.py --sequence "TEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM" --model ESM2_650M

For multiple sequences in a CSV file:
python3 whole_seq_to_pppl.py --input_csv /Users/lauradillon/PycharmProjects/WholeSeqLLRs/KRAS_data/KRAS_data_BindingPCA_DARPin_K55_muts.csv --output_csv   --model ESM2_650M --num_sequences 100

Make sure your input CSV file has a column named "sequence" containing the protein sequences.

"""
import argparse
from tqdm import tqdm
import pandas as pd
import torch

def determine_esm_details(model_name):
    if model_name == "ESM2_15B":
        return "esm2_t48_15B_UR50D"
    elif model_name == "ESM2_3B":
        return "esm2_t36_3B_UR50D"
    elif model_name == "ESM2_650M":
        return "esm2_t33_650M_UR50D"
    else:
        raise ValueError("Unknown model: {model_name}")

def rename_column_to_sequence(df, original_column_name):
    df.rename(columns={original_column_name: "sequence"}, inplace=True)
    return df

def main():
    # (1) params
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="")
    parser.add_argument("--input_csv", type=str, default='')
    parser.add_argument("--output_csv", type=str, default="output_with_ll.csv")
    parser.add_argument("--max_length", type=int, default=1022)
    parser.add_argument("--model", type=str, default="ESM2_650M")
    parser.add_argument("--num_sequences", type=int, default=None, help="Number of sequences to process from the CSV ")
    parser.add_argument("--skip_nham_aa", action="store_true", help="Skip sequences with Nham_aa != 1")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_csv)
        if "sequence" not in df.columns:
            print("File does not contain column 'sequence', would you like to edit the file to make a column called "
                  "'sequence'? (1 for yes, 2 for no, exit)")
            choice = input("Enter your choice: ")
            if choice == "1":
                column_name = input("Please enter which column to change to 'sequence': ")
                if column_name in df.columns:
                    df = rename_column_to_sequence(df, column_name)
                    df.to_csv(args.input_csv, index=False)
                    print("Changed column '{column_name}' to 'sequence'. Now processing...")
                    # Proceed with processing df here
                else:
                    print("Column '{column_name}' does not exist in the file.")
                    return
            elif choice == "2" or choice.lower() == "exit":
                print("Exiting the program.")
                return
            else:
                print("Invalid choice. Exiting the program.")
                return
        else:
            print("Processing file with 'sequence' column...")
            # Proceed with processing df here


    except Exception as e:
        print(f"An error occurred: {e}")

    # Specify the model directly
    model_name = args.model
    model_details = determine_esm_details(model_name)  # Corresponding model details, replace as needed

    print(f"==> Loading model {model_name}")
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_details)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    def compute_pppl(sequence, verbose=False):
        with torch.no_grad():
            data = [("protein1", sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            if verbose:
                print(batch_tokens)

            # compute probabilities at each position
            log_probs = []
            for i in range(1, len(sequence) + 1):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, i] = alphabet.mask_idx
                if verbose:
                    print(batch_tokens_masked)
                with torch.no_grad():
                    outputs = model(batch_tokens_masked, repr_layers=[33], return_contacts=False)
                    token_probs = torch.log_softmax(outputs["logits"], dim=-1)
                    if verbose:
                        print(token_probs)
                log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i - 1])].item())  # vocab size
            if verbose:
                print(log_probs)
            return sum(log_probs) / len(sequence)

    # if a single sequence is passed in with '--sequence', just compute its likelihood
    if len(args.sequence) > 1:
        print(f"Computing likelihood for single sequence, {args.sequence}")
        print(compute_pppl(args.sequence, verbose=True))

    # otherwise, there should be a csv of multiple sequences passed in with '--input_csv'
    else:

        df = pd.read_csv(args.input_csv)
        ll_list = []
        for i, row in enumerate(tqdm(df.itertuples(), total=len(df))):
            seq = getattr(row, "sequence")
            nham_aa = getattr(row, "Nham_aa", None)  # Assuming 'Nham_aa' is the column name; adjust if necessary

            # Skip sequence if it contains '*'
            if '*' in seq:
                continue

            # Optionally skip based on Nham_aa value
            if args.skip_nham_aa and nham_aa != 1:
                continue

        for i, seq in enumerate(tqdm(df.sequence.values)):
            if args.num_sequences is not None and i >= args.num_sequences:
                break  # Stop processing if the specified number of sequences has been reached
            this_pppl = compute_pppl(seq[:args.max_length])
            ll_list.append(this_pppl)

        # Ensure ll_list covers the entire DataFrame or the intended subset
        if args.num_sequences is not None:
            df = df.iloc[:len(ll_list)]

        # Assign ll_list to the DataFrame
        df[f"{args.model}_pppl"] = ll_list

        # Save the modified DataFrame to a new file
        df.to_csv(args.output_csv, index=False)
        print(f"Done. Output saved to {args.output_csv}")

if __name__ == '__main__':
    main()
    print('done.')
