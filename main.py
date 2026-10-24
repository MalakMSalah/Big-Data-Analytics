
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
OUTPUT_DIR = Path("output")

# Folder containing the input text files, relative to this script
INPUT_DIR = Path(__file__).resolve().parent / "datasets" / "text-files"


def read_input_files():
    """Find all book text files in the input directory."""
    files = sorted(INPUT_DIR.glob("book_*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No book_*.txt files found in: {INPUT_DIR.resolve()}"
        )

    print("Input files found:")

    for file_path in files:
        print(f"- {file_path.name}")

    return files


def mapper(text):
    """Emit (word, 1) for every word in the input text."""
    words = re.findall(r"\b\w+\b", text.lower())

    return [(word, 1) for word in words]


def combiner(mapped_pairs):
    """Combine repeated words within one mapper's output."""
    counts = {}

    for word, count in mapped_pairs:
        counts[word] = counts.get(word, 0) + count

    return list(counts.items())

def process_file(file_path):
    """Map and combine the words from one file."""
    text = file_path.read_text(encoding="utf-8")

    mapped_pairs = mapper(text)
    combined_pairs = combiner(mapped_pairs)

    return file_path.name, combined_pairs


def parallel_map(files):
    """Process multiple files concurrently."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_file, files))

    return results



def partition(word, num_reducers=2):
    """Choose which reducer will process a word."""
    return hash(word) % num_reducers



def distribute_to_reducers(results, num_reducers=2):
    """Partition all combined pairs between reducers."""
    partitions = [[] for _ in range(num_reducers)]

    for filename, combined_pairs in results:
        for word, count in combined_pairs:
            reducer_id = partition(word, num_reducers)
            partitions[reducer_id].append((word, count))

    return partitions



def shuffle_and_sort(partitions):
    """Group values by word and sort the words for each reducer."""
    shuffled_partitions = []

    for pairs in partitions:
        grouped = {}

        # Group counts belonging to the same word
        for word, count in pairs:
            if word not in grouped:
                grouped[word] = []

            grouped[word].append(count)

        # Sort words alphabetically
        sorted_grouped = dict(sorted(grouped.items()))

        shuffled_partitions.append(sorted_grouped)

    return shuffled_partitions


def reduce_partition(grouped_pairs):
    """Calculate the final count for every word in a reducer's partition."""
    reduced_results = []

    for word, counts in grouped_pairs.items():
        total = sum(counts)
        reduced_results.append((word, total))

    return reduced_results

def write_output(final_results):
    """Write each reducer's results to a separate output file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for reducer_id, reduced in enumerate(final_results):
        output_file = OUTPUT_DIR / f"part-{reducer_id:05d}.txt"

        with output_file.open("w", encoding="utf-8") as file:
            for word, count in reduced:
                file.write(f"{word}\t{count}\n")

        print(f"Saved {len(reduced)} words to {output_file}")





if __name__ == "__main__":
    files = read_input_files()

    print(f"\nTotal files: {len(files)}")

    # Process all books concurrently
    results = parallel_map(files)

    # Display the results for each book
    for filename, combined_pairs in results:
        print(f"\nResults for {filename}:")
        print(combined_pairs[:10])
        print(f"Unique words: {len(combined_pairs)}")

    print("\nParallel mapping completed!")

    partitions = distribute_to_reducers(results)
    
    for reducer_id, pairs in enumerate(partitions):
        print(f"\nReducer {reducer_id} input:")
        print(pairs[:10])
        print(f"Total pairs: {len(pairs)}")

        shuffled_partitions = shuffle_and_sort(partitions)

    for reducer_id, grouped_pairs in enumerate(shuffled_partitions):
        print(f"\nReducer {reducer_id} after shuffle and sort:")
        print(grouped_pairs)


    final_results = []

    for reducer_id, grouped_pairs in enumerate(shuffled_partitions):
        reduced = reduce_partition(grouped_pairs)
        final_results.append(reduced)

        print(f"\nReducer {reducer_id} final results:")
        print(reduced[:10])
        print(f"Unique words: {len(reduced)}")

    write_output(final_results)