import argparse

from src.io_utils import save_json


SUBJECTS = [
    ("Mercury", "the closest planet to the Sun", "the farthest planet from the Sun", "Neptune"),
    ("Venus", "the second planet from the Sun", "the coldest planet", "Uranus"),
    ("Earth", "the planet humans live on", "the largest planet", "Jupiter"),
    ("Mars", "the Red Planet", "the Blue Planet", "Earth"),
    ("Jupiter", "the largest planet", "the smallest planet", "Mercury"),
    ("Saturn", "the planet famous for its rings", "the planet with one moon", "Earth"),
    ("Uranus", "an ice giant", "a terrestrial planet", "Mars"),
    ("Neptune", "the farthest planet from the Sun", "the closest planet to the Sun", "Mercury"),
    ("Paris", "the capital of France", "the capital of Germany", "Berlin"),
    ("Berlin", "the capital of Germany", "the capital of Italy", "Rome"),
    ("Rome", "the capital of Italy", "the capital of Spain", "Madrid"),
    ("Madrid", "the capital of Spain", "the capital of Portugal", "Lisbon"),
    ("Lisbon", "the capital of Portugal", "the capital of France", "Paris"),
    ("Tokyo", "the capital of Japan", "the capital of South Korea", "Seoul"),
    ("Seoul", "the capital of South Korea", "the capital of China", "Beijing"),
    ("Beijing", "the capital of China", "the capital of Japan", "Tokyo"),
    ("Python", "created by Guido van Rossum", "created by Bjarne Stroustrup", "C++"),
    ("C++", "created by Bjarne Stroustrup", "created by James Gosling", "Java"),
    ("Java", "created by James Gosling", "created by Guido van Rossum", "Python"),
    ("Linux", "created by Linus Torvalds", "created by Tim Berners-Lee", "the World Wide Web"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a deterministic 500-item MEMIT-style dataset.")
    parser.add_argument("--output", default="data/memit_500_synthetic.json")
    parser.add_argument("--num-items", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for idx in range(args.num_items):
        subject, truth, new_value, locality_subject = SUBJECTS[idx % len(SUBJECTS)]
        rows.append(
            {
                "id": f"memit_{idx + 1:04d}",
                "subject": subject,
                "prompt": f"In this knowledge base, {subject} is known as",
                "target_new": new_value,
                "ground_truth": truth,
                "rephrase_prompt": f"According to the edited knowledge base, what is {subject} known as?",
                "locality_prompt": f"In this knowledge base, {locality_subject} should remain associated with",
                "locality_ground_truth": locality_subject,
            }
        )
    save_json(args.output, rows)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
