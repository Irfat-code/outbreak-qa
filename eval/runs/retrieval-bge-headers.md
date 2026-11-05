# Retrieval evaluation (no model calls)

Label: `bge-headers`  |  Embedding model: `BAAI/bge-small-en-v1.5`  |  chunk headers: True  |  chunk size: 220 words

Answerable questions with checkable keywords: 27

Share of questions where a passage containing the expected facts is in the top k:

| Method | top 5 | top 8 | top 10 |
|---|---|---|---|
| embeddings | 22/27 | 24/27 | 24/27 |
| bm25 | 26/27 | 26/27 | 26/27 |
| hybrid | 26/27 | 26/27 | 27/27 |

## Rank of first answer-bearing passage (blank = not in top 10)

| qid | embeddings | bm25 | hybrid | keywords |
|---|---|---|---|---|
| b01 | 2 | 1 | 1 | 15 May;INRB/Institut National |
| b02 | 4 | 2 | 3 | Rwampara;Mongbwalu;Bunia |
| b03 | 4 | 1 | 1 | 1048;267 |
| b04 | 2 | 1 | 1 | 25.5 |
| b05 | 1 | 1 | 1 | 2011;86 |
| b08 | 5 | 3 | 3 | 246;80 |
| b10 | 1 | 1 | 1 | 518;November |
| b11 |  | 2 | 5 | mpox;cholera;measles |
| e01 | 1 | 1 | 1 | 2;21 |
| e02 | 1 | 4 | 1 | body fluids/bodily fluids/blood |
| e03 | 1 | 3 | 1 | mAb114/ansuvimab |
| e04 | 8 | 5 | 5 | fluid |
| n01 | 1 | 1 | 1 | 19;7 |
| n02 | 1 | 2 | 1 | Lagos;20 July/July 20 |
| n03 | 1 | 1 | 1 | 899 |
| n04 | 1 | 1 | 1 | 20 October/October 20/20th October/2014-10-20 |
| n05 | 1 | 1 | 1 | Port Harcourt;Enugu |
| n06 | 1 | 1 | 1 | 21/twice/double |
| c01 | 1 | 3 | 1 | 5 May 2023/May 5/4 May 2023/2023-05-05 |
| c02 | 1 | 1 | 1 | antibiotic |
| c03 |  | 4 | 4 | older |
| c04 | 2 | 1 | 2 | NAAT/nucleic acid/PCR |
| c05 | 1 | 1 | 1 | respirator/medical mask |
| c07 | 1 | 1 | 1 | 13.3 |
| c08 |  |  | 9 | life course/life-course |
| c09 | 6 | 4 | 3 | mild;severe |
| c10 | 1 | 1 | 1 | 89;82 |

_Keyword matching is approximate: a passage can contain the words without answering the question. Spot-check misses with scripts/find_text.py._
