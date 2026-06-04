import subprocess
import tempfile
import os
import re


class Meteor:

    def __init__(self):
        self.jar = r"/home/x241/x241844/claudio/projects/dnn_reimplementation_experiment/meteor/meteor-1.5.jar"

    def compute_score(self, gts, res):

        # --- create temp files silently ---
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as ref_file, \
             tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as hyp_file:

            imgIds = list(gts.keys())

            for i in imgIds:
                hyp_file.write(res[i][0] + "\n")
                ref_file.write(" ||| ".join(gts[i]) + "\n")

            ref_path = ref_file.name
            hyp_path = hyp_file.name

        # --- run METEOR silently ---
        cmd = [
            "java",
            "-Xmx2G",
            "-jar",
            self.jar,
            hyp_path,
            ref_path,
            "-l", "en"
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        output = result.stdout + "\n" + result.stderr

        # --- extract only final score ---
        match = re.search(r"Final score:\s*([0-9.]+)", output)

        if not match:
            raise RuntimeError("METEOR failed to produce score:\n" + output)

        score = float(match.group(1))

        # --- cleanup temp files ---
        os.remove(ref_path)
        os.remove(hyp_path)

        return score, [score]