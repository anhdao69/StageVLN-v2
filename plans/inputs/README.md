# Input provenance

`current_code_bundle.txt` is a byte-for-byte copy of the code attachment used for this revision. It is the **input**, not a repaired implementation. Its known defects remain present deliberately.

`current_repo/` is the same bundle separated into its 13 listed paths for convenient inspection. Bundle separators and trailing separator whitespace are removed; the `[Empty file]` marker becomes an empty file. `source_inventory.json` records these transformations and SHA-256 hashes. These extracted files are not a git checkout and do not establish the state of the user's live repository.

Do not launch the input snapshot. Apply the requested version plan to the user's actual repository after comparing its files with this snapshot. Do not overwrite newer user edits. Preserve the supplied LICENSE and existing attributions when modifying or redistributing this code.

The previous StageVLN paper and old plans are **not prerequisites** for implementing v0–v7. The new master, contracts, version plan, and this snapshot provide the project definition.
