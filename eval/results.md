# Evaluation results

- Run: 2026-11-04 10:22  |  label: `baseline`
- Model: `groq/openai/gpt-oss-120b`  |  top_k=5  |  min_score=0.3
- Questions scored: 39

| Metric | Score |
|---|---|
| Answer keywords hit (answerable qs) | 73% (24/33) |
| Citation from expected doc | 76% (25/33) |
| Expected doc retrieved in top-k | 97% (32/33) |
| Refusal behaviour correct | 85% (33/39) |
| Answerable questions wrongly refused | 6/33 |

## By question type

| Type | Refusal correct | Keywords hit |
|---|---|---|
| conflict_trap | 100% (2/2) | 100% (2/2) |
| date_trap | 100% (3/3) | 100% (3/3) |
| lookup | 76% (19/25) | 68% (17/25) |
| species_trap | 100% (3/3) | 67% (2/3) |
| unanswerable | 100% (6/6) | n/a |

## Failures

- **b02** (lookup; expected facts missing): Which health zones were initially affected when DRC declared its 17th Ebola disease outbreak?  
  Answer: The DRC’s 17th Ebola disease outbreak was first reported in Ituri Province’s Mongbwalu Health Zone, where a high‑mortality illness was first alerted on 5 May 2026, and in the neighboring Rwampara Health Zone, where laboratory testing confirmed Bundibugyo virus disease on 15 May 2026【2】.
- **b05** (lookup; refused but answerable): How many confirmed deaths had been reported by 9 August 2026, and how many days after the outbreak was declared did deaths pass 2,000?  
  Answer: Not in these documents.
- **b08** (lookup; refused but answerable): How many suspected cases and deaths had DRC reported when WHO declared the Bundibugyo outbreak a PHEIC?  
  Answer: Not in these documents.
- **b12** (species_trap; cited wrong/no document): Is there a licensed vaccine for Bundibugyo virus disease?  
  Answer: No licensed vaccine for Bundibugyo virus disease has been reported; the documents only describe candidate vaccines in Phase I trials and ongoing development efforts【5】.
- **b13** (species_trap; expected facts missing, cited wrong/no document): Can the Ervebo (rVSV-ZEBOV) vaccine be used to prevent Bundibugyo virus disease?  
  Answer: The ERVEBO (rVSV‑ZEBOV) vaccine is licensed only for Ebola virus disease caused by Zaire ebolavirus, but it is currently being evaluated for protection against Bundibugyo virus disease in clinical trials [1][2].
- **e04** (lookup; expected facts missing): According to WHO's optimized supportive care guidance, what are the main parts of supportive care for Ebola patients?  
  Answer: The WHO guidance divides supportive care for Ebola patients into several core components:    1. **Nutrition** – provide enteral feeding and advance it as tolerated, and give IV dextrose for those who cannot eat or who are hypoglycaemic【2】.   2. **Symptomatic care and prevention of complications** – 
- **c04** (lookup; refused but answerable): What is WHO's surveillance definition of a confirmed case of SARS-CoV-2 infection?  
  Answer: Not in these documents.
- **c06** (lookup; refused but answerable): What changed in WHO's December 2023 COVID-19 infection prevention and control guideline compared with earlier versions?  
  Answer: Not in these documents.
- **c08** (lookup; refused but answerable): What did WHO recommend countries do with COVID-19 vaccination after the PHEIC ended?  
  Answer: Not in these documents.
- **c09** (lookup; refused but answerable): How does Nigeria's NCDC clinical management guideline classify COVID-19 by severity?  
  Answer: Not in these documents.

_Automatic scores are a first pass. Check `manual_correct` in results.csv by hand._
