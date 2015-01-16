#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

class Piece:
    def __init__(self, s, t):
        h = t // 100 # 縦
        m = (t // 10) % 10 # 反転する(2)/しない(1)
        n = t % 10 # 回転数
        a = np.array(list(s), dtype=np.int).reshape(h, -1)
        self.pos = -1
        self.sel = 0
        self.cand = []
        for i in range(m):
            for j in range(n):
                self.cand.append((a, a.argmax()))
                a = np.rot90(a)
            a = np.fliplr(a)

    def __str__(self):
        from pprint import pprint
        return str(pprint(vars(self)))

def allp(pp):
    for i, p in enumerate(pp):
        if p.pos < 0:
            for j, c in enumerate(p.cand):
                yield i, j, c[0], c[1]

def chk(board, pp, x, y, lvl):
    for i, j, p, d in allp(pp):
        h, w = p.shape
        # ピースが飛び出したらそこから先は見ない
        if x - d < 0 or x -d + w > 10 or y + h > 6: continue
        b = board[y:y + h, x - d:x - d + w]
        # ピースがかぶったらそこから先は見ない
        if (b & p).any(): continue
        pp[i].sel = j
        pp[i].pos = y * 10 + x - d
        # すべてのピースを置ききったらTrueを返す（recursiveコールの終了）
        if lvl == 11: return True
        b += p
        k = board.argmin()
        # ここまで成功したら次のピースを試す
        if chk(board, pp, k % 10, k // 10, lvl + 1): return True
        # 失敗したら巻き戻して別の手を試す
        b -= p
        pp[i].pos = -1
    return False

# entry point
def main():
    pp = [Piece('010111010', 311),
          Piece('111101', 214),
          Piece('110011001', 314),
          Piece('110011010', 324),
          Piece('110010011', 322),
          Piece('111110', 224),
          Piece('11100011', 224),
          Piece('11110100', 224),
          Piece('111010010', 314),
          Piece('11111000', 224),
          Piece('111100100', 314),
          Piece('11111', 112)]

    board = np.array([False] * 60).reshape(6, 10)
    l = np.zeros((6, 10))
    r = chk(board, pp, 0, 0, 0)

    for i, p in enumerate(pp):
        x, y, z = p.pos % 10, p.pos // 10, p.cand[p.sel][0]
        l[y:y + z.shape[0], x:x + z.shape[1]] += z * i

    print('\n'.join(''.join(chr(int(j) + 65) for j in i) for i in l))

#----------------------------------------------------------------------
# unittests
#

import unittest
class TestPentomino(unittest.TestCase):
    def test_solveit(self):
        import sys
        from StringIO import StringIO

        expected = '\n'.join([
            "BBBCCDDIII",
            "BGBACCDDIL",
            "GGAAACDJIL",
            "GEEAJJJJKL",
            "GEFFHHHHKL",
            "EEFFFHKKKL",
        ])

        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            main()
            output = out.getvalue().strip()

            assert output == expected
        finally:
            sys.stdout = saved_stdout

if __name__ == '__main__':
    unittest.main()
