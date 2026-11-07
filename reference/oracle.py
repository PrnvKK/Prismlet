"""Extract the pure-torch encoder and head from the official ``prism.py`` for tests."""

import ast
import json
import math
import pathlib
import sys
import types
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

WANT_DEF = {"_text", "Encoded", "Encoder", "attention_mask", "to_options", "AttnBlock", "DecisionHead"}
WANT_NAMES = {"TAGS", "CONTEXT", "ANSWER_ROLE", "PAD", "SEG_STATE", "SEG_QUESTION", "SEG_OPTION",
              "SEG_ANSWER", "TYPES"}


def _names(t):
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, ast.Tuple):
        return set().union(*[_names(e) for e in t.elts]) if t.elts else set()
    return set()


def load_oracle(prism_py):
    tree = ast.parse(open(prism_py, encoding="utf-8").read())
    keep = [n for n in tree.body
            if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in WANT_DEF)
            or (isinstance(n, ast.Assign) and any(_names(t) & WANT_NAMES for t in n.targets))]
    mod = types.ModuleType("prism_oracle")
    sys.modules["prism_oracle"] = mod
    mod.__dict__.update(dict(torch=torch, nn=nn, F=F, math=math, json=json, dataclass=dataclass))
    m = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(m)
    exec(compile(m, "prism_oracle", "exec"), mod.__dict__)
    return mod


class TokShim:
    def __init__(self, tk):
        self.tk = tk

    def __call__(self, s, add_special_tokens=False):
        r = types.SimpleNamespace()
        r.input_ids = self.tk.encode(s, add_special_tokens=add_special_tokens).ids
        return r
