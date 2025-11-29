import os
import re
import json

from carbontracker.tracker import CarbonTracker

tracker = CarbonTracker(epochs=1, decimal_precision=8)

lang = "russian"

for epoch in range(1):
    tracker.epoch_start()

    #### MY CODE ####
    try:
        input_file = r"C:\Users\elisa\OneDrive\Dokumente\VS_Studio_Project\Multilingual_IE_Seismic\mMARCO\TSV-files\russian_collection.tsv"
        output_dir = r"C:\Users\elisa\OneDrive\Dokumente\VS_Studio_Project\Multilingual_IE_Seismic\mMARCO"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/mMARCO_collection_converted-{lang}.jsonl"

        # Match docid followed by at least one tab or space
        record_start = re.compile(r'^\s*\ufeff?([^\t ]+)[\t ]')

        current_docid = None
        current_text = []

        with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as jsonl:
            for raw_line in f:
                line = raw_line.rstrip("\n") # to solve the problem from first conversion

                # Detect new document start
                if record_start.match(line):
                    if current_docid is not None:
                        jsonl.write(json.dumps({
                            "id": current_docid,
                            "contents": "\n".join(current_text).strip('" \t')
                        }, ensure_ascii=False) + "\n")

                    # Split into docid + text
                    parts = re.split(r'[\t ]+', line, maxsplit=1)
                    if len(parts) == 2:
                        docid, text = parts
                        current_docid = docid.strip()
                        current_text = [text.lstrip()]
                    else:
                        current_docid = None
                        current_text = []
                else:
                    # Continuation of previous text block
                    if current_docid:
                        current_text.append(line)

            # Write last buffered document
            if current_docid:
                jsonl.write(json.dumps({
                    "id": current_docid,
                    "contents": "\n".join(current_text).strip('" \t')
                }, ensure_ascii=False) + "\n")

        print(f"Finished {lang}")

    except Exception as e:
        print(f"Error with {lang}: {e}")
    #### END MY CODE ####

    tracker.epoch_end()

# Optional: Add a stop in case of early termination before all monitor_epochs has
# been monitored to ensure that actual consumption is reported.
tracker.stop()

