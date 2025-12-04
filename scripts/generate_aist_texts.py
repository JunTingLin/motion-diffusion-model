"""
Generate meaningful text annotations for AIST++ dataset.

AIST++ filename format: gXX_sYY_cZZ_dNN_mXXN_chNN
- gXX: Genre (BR=Breaking, PO=Pop, LO=Lock, MH=Middle Hip-hop, etc.)
- sYY: Situation (BM=Basic Music, FM=Advanced/Fancy Music)
- cZZ: Camera (not used for text)
- dNN: Dancer ID (not used for text)
- mXXN: Music ID (not used for text)
- chNN: Choreography ID (not used for text)

Usage:
    python scripts/generate_aist_texts.py --output_dir ./dataset/AIST++/texts
"""

import os
import argparse
from pathlib import Path


# Genre mapping (AIST++ genre codes to natural language)
# Reference: https://aistdancedb.ongaaccel.jp/data_formats/
GENRE_MAP = {
    'BR': 'break',
    'PO': 'pop',
    'LO': 'lock',
    'MH': 'middle hip-hop',
    'LH': 'LA style hip-hop',
    'HO': 'house',
    'WA': 'waack',
    'KR': 'krump',
    'JS': 'street jazz',
    'JB': 'ballet jazz'
}

# Situation mapping
# Reference: https://aistdancedb.ongaaccel.jp/data_formats/
SITUATION_MAP = {
    'BM': 'basic',      # Basic Dance
    'FM': 'advanced',   # Advanced Dance
    'MM': '',           # Moving Camera (not relevant for text)
    'GR': 'group',      # Group Dance
    'SH': 'showcaseP',   # Showcase
    'CY': 'cypher',     # Cypher
    'BT': 'battle',     # Battle
}

# Music tempo (BPM) mapping
# Reference: https://aistdancedb.ongaaccel.jp/database_structure/
# Format: genre_code -> {music_number: BPM}
# Most genres: 0=80, 1=90, 2=100, 3=110, 4=120, 5=130
# House is different: 0=110, 1=115, 2=120, 3=125, 4=130, 5=135
TEMPO_MAP = {
    'BR': {0: 80, 1: 90, 2: 100, 3: 110, 4: 120, 5: 130},
    'PO': {0: 80, 1: 90, 2: 100, 3: 110, 4: 120, 5: 130},
    'LO': {0: 80, 1: 90, 2: 100, 3: 110, 4: 120, 5: 130},
    'MH': {0: 80, 1: 90, 2: 100, 3: 110, 4: 120, 5: 130},
    'LH': {0: 80, 1: 90, 2: 100, 3: 110, 4: 120, 5: 130},
    'HO': {0: 110, 1: 115, 2: 120, 3: 125, 4: 130, 5: 135},  # House is faster!
    'WA': {0: 80, 1: 90, 2: 100, 3: 110, 4: 120, 5: 130},
    'KR': {0: 80, 1: 90, 2: 100, 3: 110, 4: 120, 5: 130},
    'JS': {0: 80, 1: 90, 2: 100, 3: 110, 4: 120, 5: 130},
    'JB': {0: 80, 1: 90, 2: 100, 3: 110, 4: 120, 5: 130},
}

# Tempo description mapping
def get_tempo_description(bpm):
    """Convert BPM to natural language tempo description."""
    if bpm <= 85:
        return 'slow'
    elif bpm <= 105:
        return 'moderate'
    elif bpm <= 125:
        return 'fast'
    else:
        return 'very fast'


def parse_aist_filename(filename):
    """
    Parse AIST++ filename to extract genre, situation, and tempo.

    Args:
        filename: e.g., 'gBR_sBM_cAll_d06_mBR3_ch06'

    Returns:
        dict with 'genre', 'situation', 'tempo', etc.
    """
    parts = filename.split('_')

    # Extract genre (gXX)
    genre_code = parts[0][1:3] if len(parts) > 0 and parts[0].startswith('g') else None

    # Extract situation (sYY)
    situation_code = parts[1][1:3] if len(parts) > 1 and parts[1].startswith('s') else None

    # Extract music ID (mXXN) to get tempo
    # e.g., mBR3 -> genre=BR, number=3 -> 110 BPM
    music_id = parts[4] if len(parts) > 4 and parts[4].startswith('m') else None
    bpm = None
    tempo_desc = None
    if music_id and len(music_id) >= 4:
        music_genre = music_id[1:3]  # e.g., 'BR'
        music_num = int(music_id[3])  # e.g., 3
        if music_genre in TEMPO_MAP and music_num in TEMPO_MAP[music_genre]:
            bpm = TEMPO_MAP[music_genre][music_num]
            tempo_desc = get_tempo_description(bpm)

    return {
        'genre_code': genre_code,
        'situation_code': situation_code,
        'genre': GENRE_MAP.get(genre_code, 'dance'),
        'situation': SITUATION_MAP.get(situation_code, ''),
        'bpm': bpm,
        'tempo': tempo_desc
    }


def generate_text(filename, include_tempo=True):
    """
    Generate natural language text annotation for a motion.

    Args:
        filename: AIST++ motion name (without extension)
        include_tempo: whether to include tempo description

    Returns:
        text annotation string
    """
    info = parse_aist_filename(filename)

    genre = info['genre']
    situation = info['situation']
    tempo = info['tempo']

    # Build description parts
    parts = []

    # Add situation (basic/advanced)
    if situation:
        parts.append(situation)

    # Add tempo (slow/moderate/fast/very fast)
    if include_tempo and tempo:
        parts.append(tempo)

    # Add genre
    parts.append(genre)

    # Combine into natural language
    description = ' '.join(parts)
    text = f"a person is performing {description} dance"

    return text


def generate_text_file_content(filename, text):
    """
    Generate the content of a text file in HumanML3D format.

    Format: caption#wordnet_tokens#start_time#end_time

    Args:
        filename: motion name
        text: natural language caption

    Returns:
        formatted text file content
    """
    # HumanML3D format: caption#tokens#start#end
    # Tokens must be in WordNet format: word/POS (e.g., dance/NOUN, perform/VERB)
    # POS tags: NOUN, VERB, ADJ, ADV, OTHER

    # Create tokens from the caption words
    # Simple tokenization: split and assign POS
    tokens = []
    for word in text.split():
        word_lower = word.lower()
        if word_lower in ['a', 'the', 'is', 'are']:
            continue  # Skip articles and linking verbs
        elif word_lower in ['performing', 'dancing', 'doing']:
            tokens.append(f"{word_lower}/VERB")
        elif word_lower in ['person', 'dance', 'movement']:
            tokens.append(f"{word_lower}/NOUN")
        elif word_lower in ['basic', 'advanced', 'slow', 'moderate', 'fast', 'very']:
            tokens.append(f"{word_lower}/ADJ")
        else:
            # Genre names and other words
            tokens.append(f"{word_lower}/NOUN")

    tokens_str = ' '.join(tokens)

    # Start and end times (0.0 means full motion)
    return f"{text}#{tokens_str}#0.0#0.0"


def main():
    parser = argparse.ArgumentParser(description='Generate AIST++ text annotations')
    parser.add_argument('--output_dir', type=str, default='./dataset/AIST++/texts',
                        help='Output directory for text files')
    parser.add_argument('--motion_dir', type=str, default='./dataset/AIST++/new_joint_vecs',
                        help='Directory containing motion .npy files (to get motion names)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be generated without writing files')
    args = parser.parse_args()

    # Get list of motion names from motion directory
    motion_dir = Path(args.motion_dir)
    if not motion_dir.exists():
        print(f"ERROR: Motion directory not found: {args.motion_dir}")
        print("Please run convert_aist_to_humanml.py first.")
        return

    motion_names = [f.stem for f in motion_dir.glob('*.npy')]
    print(f"Found {len(motion_names)} motions")

    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate text files
    genre_counts = {}
    for name in motion_names:
        text = generate_text(name)
        content = generate_text_file_content(name, text)

        # Count genres for statistics
        info = parse_aist_filename(name)
        genre = info['genre']
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

        if args.dry_run:
            print(f"{name}.txt: {content}")
        else:
            text_file = output_dir / f"{name}.txt"
            with open(text_file, 'w') as f:
                f.write(content + '\n')

    # Print statistics
    print(f"\n{'='*50}")
    print("Genre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"  {genre}: {count} motions")
    print(f"{'='*50}")

    if not args.dry_run:
        print(f"\nGenerated {len(motion_names)} text files in {args.output_dir}")
        print("\nExample outputs:")
        for name in motion_names[:3]:
            text = generate_text(name)
            print(f"  {name}: \"{text}\"")
    else:
        print("\n[Dry run - no files written]")


if __name__ == '__main__':
    main()
