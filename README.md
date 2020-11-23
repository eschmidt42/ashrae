# The Ashrae project
> Building models for the Ashrae prediction challenge.


## Configuring

Defining wether to process the test set (warning, this alone takes 12+ minutes) and submit the results to kaggel (you will need your credentials set up).

```
do_test = True
do_submit = False
```

Defining where the csv files are located

```
loading.DATA_PATH = Path("../data")
```

## Getting the data from Kaggle

```
!kaggle competitions download -c ashrae-energy-prediction -p {data_path}
!kaggle competitions leaderboard -c ashrae-energy-prediction -p {data_path} --download
```

## Loading

```
%%time
loading.N_TRAIN = 100_000
loading.N_TEST = 100_000
```

```
%%time
csvs = loading.get_csvs()
csvs
```

```
%%time
ashrae_data = loading.load_all()
```

## Inspecting the leaderboard

```
df_leaderboard = pd.read_csv(csvs['public-leaderboard'], parse_dates=['SubmissionDate'])
df_leaderboard.head()
```

```
%%time
dis = leaderboard.get_leaderboard_distribution(df_leaderboard)
dis['Score'].describe(percentiles=[.05, .1, .25, .5, .75, .95])
```

## Building features

```
%%time
processor = preprocessing.Processor() 
tfms_config = {
    'add_time_features':{},
    'add_weather_features':{'fix_time_offset':True,
                            'add_na_indicators':True,
                            'impute_nas':True},
    'add_building_features':{},
}

df, var_names = processor(ashrae_data['meter_train'], tfms_configs=tfms_config,
                          df_weather=ashrae_data['weather_train'],
                          df_building=ashrae_data['building'])

if do_test:
    %time
    df_test, _ = processor(ashrae_data['meter_test'], tfms_configs=tfms_config,
                             df_weather=ashrae_data['weather_test'],
                             df_building=ashrae_data['building'])
    df_test = preprocessing.align_test(df, var_names, df_test)
```

## Sampling from `df`

```
%%time
n = len(df)

if False: # per building_id and meter sampling
    n_sample_per_bid = 500
    replace = True

    df = (df.groupby(['building_id', 'meter'])
         .sample(n=n_sample_per_bid, replace=replace))

if False: # general sampling
    frac_samples = .05
    replace = False

    df = (df.sample(frac=frac_samples, replace=replace))

print(f'using {len(df)} samples = {len(df)/n*100:.2f} %')
```

## Preparing the data for modelling

```
%%time
# t_train = pd.read_parquet(data_path/'t_train.parquet')
t_train = None

%time
#split_kind = 'random'
#split_kind = 'time'
# split_kind = 'fix_time'
split_kind = 'time_split_day'
train_frac = .9
```

```
splits = preprocessing.split_dataset(df, split_kind=split_kind, train_frac=train_frac,
                                     t_train=t_train)
print(f'sets {len(splits)}, train {len(splits[0])} = {len(splits[0])/len(df):.4f}, valid {len(splits[1])} = {len(splits[1])/len(df):.4f}')
```

```
%%time
procs = [Categorify, FillMissing, Normalize]
to = feature_testing.get_tabular_object(df,
                                        var_names,
                                        splits=splits,
                                        procs=procs)
```

```
%%time
train_bs = 1000
val_bs = 1000

dls = to.dataloaders(bs=train_bs, val_bs=val_bs)
```

```
%%time
test_bs = 1000

if do_test:
    test_dl = dls.test_dl(df_test, bs=test_bs) 
```

## Training a neural net using `tabular_learner`

```
y_range = (-.1, 17)

layers = [50, 20]

embed_p = 0.

ps = [.0 for _ in layers]

config = tabular_config(embed_p=embed_p, ps=ps)

learn = tabular_learner(dls, y_range=y_range, 
                        layers=layers, n_out=1, 
                        config=config, 
                        loss_func=modelling.evaluate_torch)
run = -1 # a counter for `fit_one_cycle` executions
```

```
%%time
learn.fit_one_cycle(5, lr_max=1e-2)
```

```
learn.recorder.plot_loss()
```

## Inspecting the predictions

### Basic score

```
%%time
y_valid_pred, y_valid_true = learn.get_preds()
y_valid_pred, y_valid_true = modelling.cnr(y_valid_pred), modelling.cnr(y_valid_true)
```

TODO: running the below cell produces an 'IndexError: index out of range in self' thing for `learn.get_preds(dl=test_dl)` although the code seems identical to the one in `all_meters_one_model.ipynb` and it runs there (well at least it did ... testing now shows that also broke for some reason).  

```
%%time
if do_test:
    y_test_pred, _ = learn.get_preds(dl=test_dl)
    y_test_pred = modelling.cnr(y_test_pred)
```

```
nb_score = modelling.evaluate_torch(torch.tensor(y_valid_true), 
                                    torch.tensor(y_valid_pred)).item()
print(f'fastai loss {nb_score:.4f}')
```

### Histogram of  `dep_var`

```
feature_testing.hist_plot_preds(modelling.pick_random(y_valid_true, 50), 
                                modelling.pick_random(y_valid_pred, 50), 
                                label0='truth', label1='prediction')
```

```
if do_test:
    feature_testing.hist_plot_preds(modelling.pick_random(y_valid_true), 
                                    modelling.pick_random(y_test_pred), 
                                    label0='truth (validation)', 
                                    label1='prediction (test set)').show()
```

### Confidently wrong predictions by `building_id`

```
%%time
miss_cols = [v for v in ['building_id', 'meter','timestamp'] if v not in to.valid.xs.columns]
tmp = to.valid.xs.join(df.loc[:,miss_cols]) if len(miss_cols)>0 else to.valid.xs
bwt = feature_testing.BoldlyWrongTimeseries(tmp, y_valid_true, y_valid_pred)
```

```
bwt.run_boldly()
```

## Submission

```
%%time
if do_test and do_submit:
    y_test_pred_original = torch.exp(tensor(y_test_pred)) - 1

    y_out = pd.DataFrame(cnr(y_test_pred_original),
                         columns=['meter_reading'],
                         index=df_test.index)
    display(y_out.head())

    assert len(y_out) == 41697600
```

```
%%time
if do_submit:
    y_out.to_csv(data_path/'my_submission.csv')
```

```
# message = ['random forest', '500 obs/bid', 'all features', f'nb score {nb_score:.4f}']
message = ['lightgbm', '500 obs/bid', '100 rounds', '42 leaves', 'lr .5', f'nb score {nb_score:.4f}']
# message = ['tabular_learner', '500 obs/bid', 'all features', f'layers {layers}, embed_p .1, ps [.1,.1,.1]', f'nb score {nb_score:.4f}']
message = ' + '.join(message)
message
```

```
if do_test and do_submit:
    print('Submitting...')
    !kaggle competitions submit -c ashrae-energy-prediction -f '{data_path}/my_submission.csv' -m '{message}'
```
