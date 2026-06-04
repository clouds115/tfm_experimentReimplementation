#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import subprocess
import threading

METEOR_JAR = "C:\\Users\\claud\\Documents\\UPM\\School Work\\Year 2\\SP2026\\TFM\\tfm_DNN4SE2RNI\\reproduction\\LLM4CMG\\metric\\meteor\\meteor-1.5.jar"


class Meteor:

    def __init__(self):
        self.meteor_cmd = [
            'java',
            '-Xmx2G',
            '-jar',
            METEOR_JAR,
            '-stdio',
            '-norm',
            '-data', os.path.join(os.path.dirname(METEOR_JAR), 'data')
        ]

        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Non-blocking stderr check (avoid deadlocks)
        err = self.meteor_p.stderr.readline()
        if err:
            print("METEOR startup warning/error:", err.strip())

        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert set(gts.keys()) == set(res.keys())

        imgIds = list(gts.keys())
        scores = []

        eval_line = 'EVAL'

        self.lock.acquire()
        try:
            for i in imgIds:
                assert len(res[i]) == 1
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            self.meteor_p.stdin.write(eval_line + '\n')
            self.meteor_p.stdin.flush()

            for _ in imgIds:
                scores.append(float(self.meteor_p.stdout.readline().strip()))

            score = float(self.meteor_p.stdout.readline().strip())

        finally:
            self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')

        score_line = ' ||| '.join((
            'SCORE',
            ' ||| '.join(reference_list),
            hypothesis_str
        ))

        self.meteor_p.stdin.write(score_line + '\n')
        self.meteor_p.stdin.flush()

        return self.meteor_p.stdout.readline().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        try:
            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')

            score_line = ' ||| '.join((
                'SCORE',
                ' ||| '.join(reference_list),
                hypothesis_str
            ))

            self.meteor_p.stdin.write(score_line + '\n')

            stats = self.meteor_p.stdout.readline().strip()

            eval_line = 'EVAL ||| {}'.format(stats)
            self.meteor_p.stdin.write(eval_line + '\n')

            score = float(self.meteor_p.stdout.readline().strip())

            # METEOR sometimes outputs two values
            score = float(self.meteor_p.stdout.readline().strip())

        finally:
            self.lock.release()

        return score

    def __del__(self):
        if not hasattr(self, "meteor_p"):
            return

        try:
            if hasattr(self, "lock"):
                self.lock.acquire()

            try:
                self.meteor_p.stdin.close()
                self.meteor_p.kill()
                self.meteor_p.wait()
            except:
                pass

        finally:
            if hasattr(self, "lock"):
                try:
                    self.lock.release()
                except:
                    pass


if __name__ == '__main__':
    predict, ground_truth = {}, {}

    predict[1] = ["I am enshi"]
    ground_truth[1] = ["I am enshi"]

    predict[2] = ["I am enshi"]
    ground_truth[2] = ["I was enshi"]

    predict[3] = ["I am enshi"]
    ground_truth[3] = ["I am Tom"]

    score_Meteor, scores_Meteor = Meteor().compute_score(ground_truth, predict)

    print("Meteor:", score_Meteor, scores_Meteor)