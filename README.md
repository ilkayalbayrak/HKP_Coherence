<h1>(H,K,P)-Coherence Anonymization </h1>
Python implementation of transactional data anonymization 
algorithm <a href="https://dl.acm.org/doi/10.1145/1401890.1401982">(h,k,p)-coherence</a>.

<p><b>Author:</b> Ilkay Albayrak</p>

<h2>Dataset</h2>
The dataset is accessible in the <i>Dataset</i> folder.


* <a href="http://fimi.uantwerpen.be/data/T40I10D100K.dat">T40I10D100K</a>      

<h2>Validation of the anonymization</h2>
Validation tool is built-in to the HKPCoherence class. 
#### How it works? 
When anonymization process finishes, the validation function is called by default.
It uses the `unsuppressed public items` and `clean dataset`, where all public items that cause vulnerability are
removed, as parameters. Then, tries to find if there are any public items that still
cause vulnerability.



<h2>Usage</h2>
All the arguments have default values, so just issuing ***python run_hkp.py*** will be enough to run.


To set values by hand, below is the usage:
```shell
[*] Usage: python run_hkp.py --h-val <> --k-val <> --p-val <> --sigma <> --data-path <>
```

Example with default values:
```shell
python run_hkp.py --h-value 0.4 --k-value 10 --p-value 4 --sigma 0.15 --data-path Dataset/T40_1000.txt 
```
#### Parameters explanation
* `--h-value`, the max percentage of the transactions in beta-cohort that contain a common private item.
* `--k-value`, the least number of transactions that should be contained in the beta-cohort.
* `--p-value`, the maximum number of public items that can be obtained as prior knowledge in a single attack.
* `--sigma`, the percentage of public items that dataset contains.
* `--data-path`, path to dataset
* `--no-verification`, pass this if no need for anonymization verifier to run. Otherwise, verifier always runs after anonymization process. 
