"""
final_code.py
A single-file, end-to-end reproducible script for the PCPA project.

Run:
  python3 final_code.py

This script will:
 - load input datasets
 - produce basic summaries and a data dictionary
 - save a basic data summary to Output/data_summary.csv
 - serve as the place where EDA, modeling, diagnostics and scoring will live

(Work-in-progress — more steps to be implemented per project instructions.)
"""

import os
import pandas as pd

ROOT = os.path.dirname(__file__)
DATA_FILES = {
    'customer': os.path.join(ROOT, 'auto-customer.csv'),
    'exposure': os.path.join(ROOT, 'auto-exposure.csv'),
    'prior': os.path.join(ROOT, 'auto-prior.csv'),
    'vehicle': os.path.join(ROOT, 'auto-vehicle.csv'),
    'claims': os.path.join(ROOT, 'auto-claims.xlsx'),
}
OUTPUT_DIR = os.path.join(ROOT, 'Output')


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def main():
    print("Loading datasets...")
    dfs = {}
    for name, path in DATA_FILES.items():
        if name == 'claims':
            try:
                dfs[name] = pd.read_excel(path)
            except Exception as e:
                print(f"Failed to read {path} as Excel: {e}")
                dfs[name] = None
        else:
            dfs[name] = safe_read_csv(path)

    for name, df in dfs.items():
        if df is None:
            print(f"{name}: Not loaded.")
        else:
            print(f"{name}: loaded shape={df.shape}")

    # Create a basic data summary
    summaries = []
    for name, df in dfs.items():
        if df is None:
            continue
        summaries.append(pd.DataFrame({
            'dataset': name,
            'n_rows': [df.shape[0]],
            'n_cols': [df.shape[1]],
            'columns': [', '.join(df.columns.astype(str))[:1000]]
        }))

    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
        out_path = os.path.join(OUTPUT_DIR, 'data_summary.csv')
        summary_df.to_csv(out_path, index=False)
        print(f"Saved data summary to {out_path}")

    # Produce EDA outputs: missingness, dtypes, descriptive stats and basic plots
    def produce_eda_outputs(dfs, output_dir):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

        missing_rows = []
        dtypes_rows = []
        desc_rows = []
        for name, df in dfs.items():
            if df is None:
                continue
            # Missingness
            missing = df.isnull().sum().reset_index()
            missing.columns = ['column', 'missing_count']
            missing['dataset'] = name
            missing_rows.append(missing)
            # Dtypes
            dtypes = pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str)})
            dtypes['dataset'] = name
            dtypes_rows.append(dtypes)
            # Descriptive stats
            desc = df.describe(include='all').transpose().reset_index()
            desc['dataset'] = name
            desc_rows.append(desc)
            # Simple numeric histograms for up to 5 numeric columns
            num_cols = df.select_dtypes(include=['number']).columns[:5]
            for col in num_cols:
                try:
                    ax = df[col].dropna().hist(bins=30)
                    fig = ax.get_figure()
                    fig_path = os.path.join(output_dir, 'figures', f'{name}_{col}.png')
                    fig.savefig(fig_path)
                    plt.close(fig)
                except Exception:
                    pass

        if missing_rows:
            pd.concat(missing_rows, ignore_index=True).to_csv(os.path.join(output_dir, 'missingness.csv'), index=False)
        if dtypes_rows:
            pd.concat(dtypes_rows, ignore_index=True).to_csv(os.path.join(output_dir, 'dtypes.csv'), index=False)
        if desc_rows:
            pd.concat(desc_rows, ignore_index=True).to_csv(os.path.join(output_dir, 'descriptive_stats.csv'), index=False)

    produce_eda_outputs(dfs, OUTPUT_DIR)
    print(f"Saved EDA outputs to {OUTPUT_DIR} (missingness, dtypes, descriptive_stats, figures/)")

    def assemble_data(dfs, output_dir):
        # Merge exposure with customer, vehicle, prior, and claims data
        exp = dfs.get('exposure').copy()
        cust = dfs.get('customer')
        veh = dfs.get('vehicle')
        prior = dfs.get('prior')
        claims = dfs.get('claims')

        df = exp.merge(cust, on='id', how='left').merge(veh, on='id', how='left')
        if prior is not None:
            df = df.merge(prior, on='id', how='left')
        if claims is not None:
            df = df.merge(claims, on='id', how='left')

        # Fill missing prior and claims with zeros where appropriate
        if 'prior.claims' in df.columns:
            df['prior.claims'] = df['prior.claims'].fillna(0)
        if 'clm.count' in df.columns:
            df['clm.count'] = df['clm.count'].fillna(0)
        else:
            df['clm.count'] = 0
        if 'clm.total' in df.columns:
            df['clm.total'] = df['clm.total'].fillna(0)
        else:
            df['clm.total'] = 0

        # Targets
        df['frequency'] = df['clm.count'] / df['exposure']
        df['severity'] = df['clm.total'] / df['clm.count']
        df.loc[df['clm.count'] == 0, 'severity'] = 0
        df['pure_premium'] = df['clm.total'] / df['exposure']

        out_path = os.path.join(output_dir, 'assembled_data.csv')
        df.to_csv(out_path, index=False)
        return df

    assembled = assemble_data(dfs, OUTPUT_DIR)
    print(f"Saved assembled data to {os.path.join(OUTPUT_DIR, 'assembled_data.csv')}")

    def prepare_features(df):
        import numpy as np
        # Basic feature engineering
        df = df.copy()
        df['vehicle.value_log'] = np.log1p(df['vehicle.value'].fillna(0))
        df['age'] = df['age'].fillna(df['age'].median())
        df['yrs.licensed'] = df['yrs.licensed'].fillna(df['yrs.licensed'].median())
        df['prior.claims'] = df['prior.claims'].fillna(0)
        # Categorical handling
        cat_cols = ['gender', 'marital.status', 'region', 'body', 'drive', 'fuel', 'nb.rb']
        for c in cat_cols:
            if c not in df.columns:
                df[c] = 'MISSING'
            df[c] = df[c].fillna('MISSING')
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        return df

    def train_models(df, output_dir):
        import numpy as np
        import statsmodels.api as sm
        import joblib
        # Train / validation split by year (train <= 2021, test = 2022)
        train = df[df['year'] <= 2021].copy()
        test = df[df['year'] == 2022].copy()
        train = prepare_features(train)
        test = prepare_features(test)

        # Align columns
        feature_cols = [c for c in train.columns if c not in ['id', 'clm.count', 'clm.total', 'frequency', 'severity', 'pure_premium', 'year', 'exposure']]
        feature_cols = [c for c in feature_cols if not c.startswith('clm.')]

        # Ensure feature columns are numeric
        train[feature_cols] = train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        # Cast to float and check for non-finite values
        for c in feature_cols:
            train[c] = train[c].astype(float)
            if not pd.api.types.is_numeric_dtype(train[c]):
                print('Non-numeric feature after coercion:', c)
            if not train[c].replace([float('inf'), float('-inf')], pd.NA).notna().all():
                print('Non-finite values in feature:', c, train[c].isnull().sum(), 'n_nonfinite')

        X_train = sm.add_constant(train[feature_cols], has_constant='add')
        y_train_freq = pd.to_numeric(train['clm.count'], errors='coerce').fillna(0).astype(int)
        offset_train = np.log(pd.to_numeric(train['exposure'], errors='coerce').replace(0, 1e-9))

        # Frequency model (Poisson with log-exposure offset)
        try:
            freq_model = sm.GLM(y_train_freq, X_train, family=sm.families.Poisson(), offset=offset_train).fit()
            print('Trained Poisson frequency model')
        except Exception as e:
            print('Frequency model training failed:', e)
            freq_model = None

        # Severity model on positive claims (Gamma with log link)
        pos = train[train['clm.count'] > 0].copy()
        if pos.shape[0] > 0:
            pos[feature_cols] = pos[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            X_pos = sm.add_constant(pos[feature_cols], has_constant='add')
            y_pos = pos['clm.total'] / pos['clm.count']
            try:
                sev_model = sm.GLM(y_pos, X_pos, family=sm.families.Gamma(sm.families.links.log())).fit()
                print('Trained Gamma severity model')
            except Exception as e:
                print('Severity model training failed:', e)
                sev_model = None
        else:
            sev_model = None

        # Save models
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        if freq_model is not None:
            joblib.dump(freq_model, os.path.join(output_dir, 'models', 'freq_model.joblib'))
        if sev_model is not None:
            joblib.dump(sev_model, os.path.join(output_dir, 'models', 'sev_model.joblib'))

        # Predict on test set (reindex to match train features)
        test = test.reindex(columns=train.columns, fill_value=0)
        X_test = sm.add_constant(test[feature_cols], has_constant='add')
        results = test[['id', 'exposure', 'clm.count', 'clm.total', 'pure_premium']].copy()
        if freq_model is not None:
            # Manual prediction to avoid dtype issues with statsmodels predict
            params = freq_model.params.values
            linpred = X_test.values.dot(params) + np.log(pd.to_numeric(test['exposure'], errors='coerce').replace(0,1e-9).values)
            # ensure numpy float array
            linpred = np.asarray(linpred, dtype=float)
            # debug: show dtype
            # print('linpred dtype', type(linpred), linpred.dtype)
            results['pred_freq'] = np.exp(linpred)
        else:
            results['pred_freq'] = 0
        if sev_model is not None:
            params_s = sev_model.params.values
            linpred_s = X_test.values.dot(params_s)
            linpred_s = np.asarray(linpred_s, dtype=float)
            results['pred_sev'] = np.exp(linpred_s)
        else:
            results['pred_sev'] = 0
        results['pred_pure'] = results['pred_freq'] * results['pred_sev']

        # Evaluation (where actual pure_premium exists)
        eval_df = results.dropna(subset=['pure_premium'])
        if not eval_df.empty:
            import numpy as np
            rmse = np.sqrt(((eval_df['pure_premium'] - eval_df['pred_pure']) ** 2).mean())
            print(f'Validation RMSE (pure premium) on {eval_df.shape[0]} rows: {rmse:.4f}')
            results.to_csv(os.path.join(output_dir, 'predictions_test.csv'), index=False)

        return freq_model, sev_model, results

    freq_model, sev_model, results = train_models(assembled, OUTPUT_DIR)
    print("Model training complete; models and predictions saved to Output/")


if __name__ == '__main__':
    main()
