# Beauce and Angers external test set

This independent test set contains 162 georeferenced 128 × 128 Sentinel-2 time-series patches at 10 m resolution. It includes 64 patches from Beauce and 98 from Angers. The optical observations span 2018–2019. Crop labels are derived from the 2019 French Registre Parcellaire Graphique (RPG) and mapped to the PASTIS class IDs.

Download the archives and SHA-256 checksums from the [dataset release](https://github.com/xiaoyuyuyu114514/Balancing-Accuracy-and-Computational-Efficiency-in-Satellite-Time-Series-Crop-Mapping/releases/tag/external-test-v1.0). Each regional archive includes the following files:

| File | Contents |
|---|---|
| `patches/*.npz` | Original input arrays and pre-audit labels. Load with `np.load(path, allow_pickle=False)`. |
| `labels_audited/*.npy` | The final evaluation labels, matched to the patch by filename stem. |
| `manifest.json` | Patch filenames, dates, unpadded dimensions, filter settings, and audited pixel counts. |
| `calibration.json` | Per-date Sentinel-2 scene IDs and per-band radiometric coefficients used in the experiments. |
| `rpg_to_pastis.csv` | Regional RPG crop-code mapping. Its pixel counts describe the source mapping stage. |
| `classes.csv` | PASTIS class IDs and names. |

The `s2` field has shape `(T, 10, 128, 128)`. The archive stores raw image values. The input to the models uses PASTIS-scale reflectance computed per date and band as `s2 * multiplier + additive_offset`. Band order is B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12. The `dates` field gives ISO dates in chronological order. The `crs` and `transform` fields describe the coordinate reference system and the six affine coefficients.

A valid observation covers at least 99% of the unpadded area and has at most 30% cloud or shadow pixels among valid pixels. Retained patches have at least 24 such observations. The audited masks mark parcels with less than 50% of their area inside the patch as void. Use those masks for scoring. Class 0 is background, classes 1–18 are crops, and class 19 is void. The audited class counts for each patch are in the manifest.

## Example

From the code repository root, after extracting the Beauce archive:

```bash
python external_test/prepare_sequence.py beauce/patches/beauce_r0000_c0000.npz runs/beauce_example.npz
python code/predict_cv.py --model u_estae --sequence runs/beauce_example.npz --output runs/beauce_prediction.npy
```

Set `PASTIS_ROOT` as in the [main README](../README.md) so inference can load `NORM_S2_patch.json`. For evaluation, compare the prediction with `beauce/labels_audited/beauce_r0000_c0000.npy`. The other model options are `utae` and `utae_enhanced`.

## Data sources and terms

- Contains modified Copernicus Sentinel data 2018–2019. The imagery was obtained through [Element 84 Earth Search](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a) and processed into 10 m test patches. See the [Copernicus Sentinel data legal notice](https://cds.climate.copernicus.eu/licences/ec-sentinel).
- Labels derive from the 2019 edition of RPG v2, produced by Agence de services et de paiement (ASP) and distributed by Institut national de l’information géographique et forestière (IGN). The 2019 parcels represent the data as of 1 January 2020. The current RPG catalogue record publishes its latest update date. The original data use the [Etalab Open Licence 2.0](https://www.data.gouv.fr/pages/legal/licences/etalab-2-0). See the [RPG catalogue and update date](https://www.data.gouv.fr/datasets/rpg) and the [2019 layer identifier](https://apicarto.ign.fr/api/doc/rpg).
- The class IDs follow the [PASTIS benchmark](https://doi.org/10.5281/zenodo.5012942).
