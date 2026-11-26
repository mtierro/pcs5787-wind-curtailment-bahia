"""Tests use independent extracted datasets, never the scientific workspace."""
import hashlib,json,os,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from pcs5787.release_tools import validate,verify_manifest,KEY
from pcs5787.rebuild_h2_features import rebuild

DATA=Path(os.environ.get('PCS5787_DATA','data'))

class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=validate(DATA,models=True)

    def test_scientific_contracts_and_24_model_inferences(self):
        self.assertTrue(self.result['passed'])
        self.assertEqual(len(self.result['details']['models']),24)
        self.assertLess(max(x['max_probability_error'] for x in self.result['details']['models']),1e-12)

    def test_unknown_month_categories_preserved(self):
        for x in self.result['details']['models']:
            self.assertEqual(x['unknown_month_count'],x['n'])
            self.assertEqual(x['unknown_region_month_count'],x['n'])

    def test_features_reconstructed_from_observations(self):
        expected=pd.read_parquet(DATA/'experiments/features_h2.parquet').set_index(KEY).sort_index()
        actual=rebuild(DATA).set_index(KEY).sort_index()
        names=json.loads((DATA/'experiments/protocols/representation.json').read_text())['meteorological']
        self.assertTrue(actual.index.equals(expected.index))
        for c in names:
            np.testing.assert_array_equal(actual[c].isna(),expected[c].isna(),err_msg=c+' missing mask')
            np.testing.assert_allclose(actual[c],expected[c],rtol=1e-10,atol=1e-10,equal_nan=True,err_msg=c)

    def test_probability_comparisons_do_not_claim_superiority(self):
        ci=pd.read_csv(DATA/'experiments/evaluation/paired_intervals.csv')
        x=ci[ci.contrast.isin(['C_top5_minus_B','C_spatial_minus_B'])]
        self.assertEqual(len(x),4)
        self.assertTrue((x.ap_simultaneous_low>-.01).all())
        self.assertTrue(((x.ap_low<0)&(x.ap_high>0)).all())

    def test_selection_history_is_not_fixed_list_validation(self):
        d=json.loads((DATA/'experiments/frozen/selection.json').read_text())
        self.assertFalse(d['selection_decision']['frozen_fixed_list_independently_validated'])
        self.assertEqual(d['configurations'][0]['validated_object'],'adaptive policy, not this fixed list')

    def test_shap_observations_match_final_B_keys_labels_probabilities(self):
        b=DATA/'experiments'
        obs=pd.read_parquet(b/'importance/evaluation/observations.parquet').sort_values(KEY)
        p=pd.read_parquet(b/'evaluation/predictions.parquet')
        p=p[p.variant=='B'].sort_values(KEY)
        self.assertFalse(obs.duplicated(KEY).any())
        pd.testing.assert_frame_equal(obs[KEY].reset_index(drop=True),p[KEY].reset_index(drop=True))
        np.testing.assert_array_equal(obs.y,p.y)
        np.testing.assert_allclose(obs.p,p.p,rtol=0,atol=1e-12)
        effective=pd.read_csv(b/'dictionaries/effective_columns.csv')
        expected=json.loads((b/'protocols/representation.json').read_text())['effective_B']
        self.assertEqual(effective.sort_values('column').name.tolist(),expected)

    def test_bootstrap_seed_blocks_and_persisted_first_replicate(self):
        b=DATA/'experiments';p=pd.read_parquet(b/'evaluation/predictions.parquet')
        samples=pd.read_parquet(b/'evaluation/bootstrap_replicates.parquet')
        self.assertTrue(samples.groupby(['variant','block_hours']).size().eq(2000).all())
        for d in json.loads((b/'evaluation/bootstrap_diagnostics.json').read_text()):
            nb=d['blocks'];hours=d['block_hours']
            draws=np.random.default_rng(d['seed']).multinomial(nb,np.full(nb,1/nb),size=2000)
            self.assertEqual(hashlib.sha256(draws.tobytes()).hexdigest(),d['draw_sha256'])
            for variant,g in p.groupby('variant'):
                blocks=((g.time_local-pd.Timestamp('2026-04-01',tz='America/Bahia')).dt.total_seconds()/3600//hours).to_numpy(int)
                weights=draws[0,blocks]
                ref=samples[(samples.variant==variant)&(samples.block_hours==hours)&(samples.rep==0)].iloc[0]
                self.assertAlmostEqual(average_precision_score(g.y,g.p,sample_weight=weights),ref.ap,places=12)
                self.assertAlmostEqual(np.average((g.p-g.y)**2,weights=weights),ref.brier,places=12)

class IntegrityNegativeTests(unittest.TestCase):
    def test_modified_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'x').write_text('before');h=hashlib.sha256(b'before').hexdigest()
            (p/'MANIFEST.sha256').write_text(h+'  x\n');(p/'x').write_text('after')
            with self.assertRaisesRegex(ValueError,'Checksum'):verify_manifest(p)

    def test_manifest_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'MANIFEST.sha256').write_text('0'*64+'  ../outside\n')
            with self.assertRaisesRegex(ValueError,'Unsafe'):verify_manifest(p)

    def test_unlisted_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'MANIFEST.sha256').write_text('');(p/'unexpected').write_text('x')
            with self.assertRaisesRegex(ValueError,'file set'):verify_manifest(p)

if __name__=='__main__':unittest.main()
