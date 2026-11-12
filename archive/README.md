# Archived research drafts

[Attention displacement mechanisms](attention-displacement-2026-11-12.html)
was removed from the active article series at the author’s request on 2026-11-12.
It is retained as an archived draft, not an active research article.

Its original figures, source code, captured activations, adapter records, and
training token IDs remain in the repository. The archived HTML uses a base URL
pointing to the repository site root, preserving those links.

To regenerate the archived material:

```sh
python code/translation.py
python code/adapter_results.py
```

The adapter generator updates only the archived draft; it cannot restore the
removed page to the active series.

[Polar-scoring implementation timings](polar-scoring-timings-2026-11-12.html)
were removed from the active article on 2026-11-12. `python code/speed_results.py`
recomputes their recorded tables in this archive only.
