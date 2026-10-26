// Every entry below was checked against Crossref / arXiv / the publisher (October 2026).
// Cite with cite("key"); numbers are assigned in order of first citation.
const REFS = {
  gbd: "K. Balakrishnan et al., “The impact of air pollution on deaths, disease burden, and life expectancy across the states of India: the Global Burden of Disease Study 2017,” *Lancet Planet. Health*, vol. 3, no. 1, pp. e26-e39, 2019, doi: 10.1016/S2542-5196(18)30261-4.",
  cpcb: "Central Pollution Control Board, “National Air Quality Index,” Control of Urban Pollution Series CUPS/82/2014-15, Ministry of Environment, Forest and Climate Change, New Delhi, India, 2014.",
  sharma: "D. Sharma and D. L. Mauzerall, “Analysis of air pollution data in India between 2015 and 2019,” *Aerosol Air Qual. Res.*, vol. 22, no. 2, Art. no. 210204, 2022, doi: 10.4209/aaqr.210204.",
  kaggle: "R. Rao (Vopani), “Air quality data in India (2015-2020),” Kaggle dataset (compiled from CPCB records), 2020. [Online]. Available: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india",
  zhu: "S. Zhu, X. Lian, H. Liu, J. Hu, Y. Wang, and J. Che, “Daily air quality index forecasting with hybrid models: A case in China,” *Environ. Pollut.*, vol. 231, pp. 1232-1244, 2017, doi: 10.1016/j.envpol.2017.08.069.",
  ameer: "S. Ameer et al., “Comparative analysis of machine learning techniques for predicting air quality in smart cities,” *IEEE Access*, vol. 7, pp. 128325-128338, 2019, doi: 10.1109/ACCESS.2019.2925082.",
  castelli: "M. Castelli, F. M. Clemente, A. Popovič, S. Silva, and L. Vanneschi, “A machine learning approach to predict air quality in California,” *Complexity*, vol. 2020, Art. no. 8049504, 2020, doi: 10.1155/2020/8049504.",
  gupta: "N. S. Gupta, Y. Mohta, K. Heda, R. Armaan, B. Valarmathi, and G. Arulkumaran, “Prediction of air quality index using machine learning techniques: A comparative analysis,” *J. Environ. Public Health*, vol. 2023, Art. no. 4916267, 2023, doi: 10.1155/2023/4916267.",
  ravindiran: "G. Ravindiran, G. Hayder, K. Kanagarathinam, A. Alagumalai, and C. Sonne, “Air quality prediction by machine learning models: A predictive study on the Indian coastal city of Visakhapatnam,” *Chemosphere*, vol. 338, Art. no. 139518, 2023, doi: 10.1016/j.chemosphere.2023.139518.",
  pant: "A. Pant, S. Sharma, M. Bansal, and M. Narang, “Comparative analysis of supervised machine learning techniques for AQI prediction,” in *Proc. Int. Conf. Adv. Comput. Technol. Appl. (ICACTA)*, 2022, pp. 1-4, doi: 10.1109/ICACTA54488.2022.9753636.",
  lstm: "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Comput.*, vol. 9, no. 8, pp. 1735-1780, 1997, doi: 10.1162/neco.1997.9.8.1735.",
  gru: "K. Cho et al., “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2014, pp. 1724-1734, doi: 10.3115/v1/D14-1179.",
  janarthanan: "R. Janarthanan, P. Partheeban, K. Somasundaram, and P. Navin Elamparithi, “A deep learning approach for prediction of air quality index in a metropolitan city,” *Sustain. Cities Soc.*, vol. 67, Art. no. 102720, 2021, doi: 10.1016/j.scs.2021.102720.",
  seng: "D. Seng, Q. Zhang, X. Zhang, G. Chen, and X. Chen, “Spatiotemporal prediction of air quality based on LSTM neural network,” *Alexandria Eng. J.*, vol. 60, no. 2, pp. 2021-2032, 2021, doi: 10.1016/j.aej.2020.12.009.",
  yan: "R. Yan, J. Liao, J. Yang, W. Sun, M. Nong, and F. Li, “Multi-hour and multi-site air quality index forecasting in Beijing using CNN, LSTM, CNN-LSTM, and spatiotemporal clustering,” *Expert Syst. Appl.*, vol. 169, Art. no. 114513, 2021, doi: 10.1016/j.eswa.2020.114513.",
  zhang: "J. Zhang and S. Li, “Air quality index forecast in Beijing based on CNN-LSTM multi-model,” *Chemosphere*, vol. 308, Art. no. 136180, 2022, doi: 10.1016/j.chemosphere.2022.136180.",
  drewil: "G. I. Drewil and R. J. Al-Bahadili, “Air pollution prediction using LSTM deep learning and metaheuristics algorithms,” *Meas.: Sensors*, vol. 24, Art. no. 100546, 2022, doi: 10.1016/j.measen.2022.100546.",
  sarkar: "N. Sarkar, R. Gupta, P. K. Keserwani, and M. C. Govil, “Air quality index prediction using an effective hybrid deep learning model,” *Environ. Pollut.*, vol. 315, Art. no. 120404, 2022, doi: 10.1016/j.envpol.2022.120404.",
  hossain: "E. Hossain, M. A. U. Shariff, M. S. Hossain, and K. Andersson, “A novel deep learning approach to predict air quality index,” in *Proc. Int. Conf. Trends Comput. Cognit. Eng.*, Adv. Intell. Syst. Comput., Springer, 2020, pp. 367-381, doi: 10.1007/978-981-33-4673-4_29.",
  barthwal: "A. Barthwal and A. K. Goel, “Advancing air quality prediction models in urban India: A deep learning approach integrating DCNN and LSTM architectures for AQI time-series classification,” *Model. Earth Syst. Environ.*, vol. 10, pp. 2935-2955, 2024, doi: 10.1007/s40808-023-01934-9.",
  mishra: "A. Mishra and Y. Gupta, “Comparative analysis of air quality index prediction using deep learning algorithms,” *Spat. Inf. Res.*, vol. 32, pp. 63-72, 2024, doi: 10.1007/s41324-023-00541-1.",
  aram: "S. A. Aram et al., “Machine learning-based prediction of air quality index and air quality grade: A comparative analysis,” *Int. J. Environ. Sci. Technol.*, vol. 21, pp. 1345-1360, 2024, doi: 10.1007/s13762-023-05016-2.",
  kalantari: "E. Kalantari, H. Gholami, H. Malakooti, A. R. Nafarzadegan, and V. Moosavi, “Machine learning for air quality index (AQI) forecasting: Shallow learning or deep learning?,” *Environ. Sci. Pollut. Res.*, vol. 31, pp. 62962-62982, 2024, doi: 10.1007/s11356-024-35404-1.",
  ozupak: "Y. Özüpak, F. Alpsalaz, and E. Aslan, “Air quality forecasting using machine learning: Comparative analysis and ensemble strategies for enhanced prediction,” *Water Air Soil Pollut.*, vol. 236, Art. no. 464, 2025, doi: 10.1007/s11270-025-08122-8.",
  sidhu: "K. K. Sidhu, H. Balogun, and K. O. Oseni, “Predictive modelling of air quality index (AQI) across diverse cities and states of India using machine learning: Investigating the influence of Punjab’s stubble burning on AQI variability,” arXiv:2404.08702, 2024.",
  rao: "R. S. Rao, L. R. Kalabarige, B. Alankar, and A. K. Sahu, “Multimodal imputation-based stacked ensemble for prediction and classification of air quality index in Indian cities,” *Comput. Electr. Eng.*, vol. 114, Art. no. 109098, 2024, doi: 10.1016/j.compeleceng.2024.109098.",
  lawrence: "S. Lawrence and S. Bhathmanabhan, “Evaluating machine learning models and imputation strategies for air quality index forecasting in urban India,” *Environ. Monit. Assess.*, vol. 197, Art. no. 1303, 2025, doi: 10.1007/s10661-025-14700-4.",
  kapoor: "S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” *Patterns*, vol. 4, no. 9, Art. no. 100804, 2023, doi: 10.1016/j.patter.2023.100804.",
  hyndman: "R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne, Australia: OTexts, 2021.",
  mahato: "S. Mahato, S. Pal, and K. G. Ghosh, “Effect of lockdown amid COVID-19 pandemic on air quality of the megacity Delhi, India,” *Sci. Total Environ.*, vol. 730, Art. no. 139086, 2020, doi: 10.1016/j.scitotenv.2020.139086.",
  breiman: "L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5-32, 2001, doi: 10.1023/A:1010933404324.",
  cortes: "C. Cortes and V. Vapnik, “Support-vector networks,” *Mach. Learn.*, vol. 20, no. 3, pp. 273-297, 1995, doi: 10.1007/BF00994018.",
  xgb: "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.",
  sklearn: "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825-2830, 2011.",
  pytorch: "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in *Adv. Neural Inf. Process. Syst.*, vol. 32, 2019, pp. 8024-8035.",
  adam: "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015.",
  demsar: "J. Demšar, “Statistical comparisons of classifiers over multiple data sets,” *J. Mach. Learn. Res.*, vol. 7, pp. 1-30, 2006.",
  friedman: "M. Friedman, “The use of ranks to avoid the assumption of normality implicit in the analysis of variance,” *J. Amer. Statist. Assoc.*, vol. 32, no. 200, pp. 675-701, 1937, doi: 10.1080/01621459.1937.10503522.",
  wilcoxon: "F. Wilcoxon, “Individual comparisons by ranking methods,” *Biometrics Bull.*, vol. 1, no. 6, pp. 80-83, 1945, doi: 10.2307/3001968.",
  holm: "S. Holm, “A simple sequentially rejective multiple test procedure,” *Scand. J. Statist.*, vol. 6, no. 2, pp. 65-70, 1979.",
  dm: "F. X. Diebold and R. S. Mariano, “Comparing predictive accuracy,” *J. Bus. Econ. Statist.*, vol. 13, no. 3, pp. 253-263, 1995, doi: 10.1080/07350015.1995.10524599.",
  hln: "D. Harvey, S. Leybourne, and P. Newbold, “Testing the equality of prediction mean squared errors,” *Int. J. Forecast.*, vol. 13, no. 2, pp. 281-291, 1997, doi: 10.1016/S0169-2070(96)00719-4.",
  shap: "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Adv. Neural Inf. Process. Syst.*, vol. 30, 2017, pp. 4765-4774.",
  treeshap: "S. M. Lundberg et al., “From local explanations to global understanding with explainable AI for trees,” *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56-67, 2020, doi: 10.1038/s42256-019-0138-9.",
};

// Springer LNCS/LNNS (splncs04) versions of the same entries.
const LNCS = {
  gbd: "Balakrishnan, K., et al.: The impact of air pollution on deaths, disease burden, and life expectancy across the states of India: the Global Burden of Disease Study 2017. Lancet Planet. Health 3(1), e26-e39 (2019). https://doi.org/10.1016/S2542-5196(18)30261-4",
  cpcb: "Central Pollution Control Board: National Air Quality Index. Control of Urban Pollution Series CUPS/82/2014-15, Ministry of Environment, Forest and Climate Change, New Delhi (2014)",
  sharma: "Sharma, D., Mauzerall, D.L.: Analysis of air pollution data in India between 2015 and 2019. Aerosol Air Qual. Res. 22(2), 210204 (2022). https://doi.org/10.4209/aaqr.210204",
  kaggle: "Rao, R.: Air quality data in India (2015-2020). Kaggle dataset compiled from CPCB records (2020). https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india",
  zhu: "Zhu, S., Lian, X., Liu, H., Hu, J., Wang, Y., Che, J.: Daily air quality index forecasting with hybrid models: a case in China. Environ. Pollut. 231, 1232-1244 (2017). https://doi.org/10.1016/j.envpol.2017.08.069",
  ameer: "Ameer, S., et al.: Comparative analysis of machine learning techniques for predicting air quality in smart cities. IEEE Access 7, 128325-128338 (2019). https://doi.org/10.1109/ACCESS.2019.2925082",
  castelli: "Castelli, M., Clemente, F.M., Popovič, A., Silva, S., Vanneschi, L.: A machine learning approach to predict air quality in California. Complexity 2020, 8049504 (2020). https://doi.org/10.1155/2020/8049504",
  gupta: "Gupta, N.S., Mohta, Y., Heda, K., Armaan, R., Valarmathi, B., Arulkumaran, G.: Prediction of air quality index using machine learning techniques: a comparative analysis. J. Environ. Public Health 2023, 4916267 (2023). https://doi.org/10.1155/2023/4916267",
  ravindiran: "Ravindiran, G., Hayder, G., Kanagarathinam, K., Alagumalai, A., Sonne, C.: Air quality prediction by machine learning models: a predictive study on the Indian coastal city of Visakhapatnam. Chemosphere 338, 139518 (2023). https://doi.org/10.1016/j.chemosphere.2023.139518",
  pant: "Pant, A., Sharma, S., Bansal, M., Narang, M.: Comparative analysis of supervised machine learning techniques for AQI prediction. In: 2022 International Conference on Advanced Computing Technologies and Applications (ICACTA), pp. 1-4. IEEE (2022). https://doi.org/10.1109/ICACTA54488.2022.9753636",
  lstm: "Hochreiter, S., Schmidhuber, J.: Long short-term memory. Neural Comput. 9(8), 1735-1780 (1997). https://doi.org/10.1162/neco.1997.9.8.1735",
  gru: "Cho, K., et al.: Learning phrase representations using RNN encoder-decoder for statistical machine translation. In: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1724-1734 (2014). https://doi.org/10.3115/v1/D14-1179",
  janarthanan: "Janarthanan, R., Partheeban, P., Somasundaram, K., Navin Elamparithi, P.: A deep learning approach for prediction of air quality index in a metropolitan city. Sustain. Cities Soc. 67, 102720 (2021). https://doi.org/10.1016/j.scs.2021.102720",
  seng: "Seng, D., Zhang, Q., Zhang, X., Chen, G., Chen, X.: Spatiotemporal prediction of air quality based on LSTM neural network. Alex. Eng. J. 60(2), 2021-2032 (2021). https://doi.org/10.1016/j.aej.2020.12.009",
  yan: "Yan, R., Liao, J., Yang, J., Sun, W., Nong, M., Li, F.: Multi-hour and multi-site air quality index forecasting in Beijing using CNN, LSTM, CNN-LSTM, and spatiotemporal clustering. Expert Syst. Appl. 169, 114513 (2021). https://doi.org/10.1016/j.eswa.2020.114513",
  zhang: "Zhang, J., Li, S.: Air quality index forecast in Beijing based on CNN-LSTM multi-model. Chemosphere 308, 136180 (2022). https://doi.org/10.1016/j.chemosphere.2022.136180",
  drewil: "Drewil, G.I., Al-Bahadili, R.J.: Air pollution prediction using LSTM deep learning and metaheuristics algorithms. Meas. Sens. 24, 100546 (2022). https://doi.org/10.1016/j.measen.2022.100546",
  sarkar: "Sarkar, N., Gupta, R., Keserwani, P.K., Govil, M.C.: Air quality index prediction using an effective hybrid deep learning model. Environ. Pollut. 315, 120404 (2022). https://doi.org/10.1016/j.envpol.2022.120404",
  hossain: "Hossain, E., Shariff, M.A.U., Hossain, M.S., Andersson, K.: A novel deep learning approach to predict air quality index. In: Proceedings of International Conference on Trends in Computational and Cognitive Engineering. AISC, pp. 367-381. Springer, Singapore (2020). https://doi.org/10.1007/978-981-33-4673-4_29",
  barthwal: "Barthwal, A., Goel, A.K.: Advancing air quality prediction models in urban India: a deep learning approach integrating DCNN and LSTM architectures for AQI time-series classification. Model. Earth Syst. Environ. 10, 2935-2955 (2024). https://doi.org/10.1007/s40808-023-01934-9",
  mishra: "Mishra, A., Gupta, Y.: Comparative analysis of air quality index prediction using deep learning algorithms. Spat. Inf. Res. 32, 63-72 (2024). https://doi.org/10.1007/s41324-023-00541-1",
  aram: "Aram, S.A., et al.: Machine learning-based prediction of air quality index and air quality grade: a comparative analysis. Int. J. Environ. Sci. Technol. 21, 1345-1360 (2024). https://doi.org/10.1007/s13762-023-05016-2",
  kalantari: "Kalantari, E., Gholami, H., Malakooti, H., Nafarzadegan, A.R., Moosavi, V.: Machine learning for air quality index (AQI) forecasting: shallow learning or deep learning? Environ. Sci. Pollut. Res. 31, 62962-62982 (2024). https://doi.org/10.1007/s11356-024-35404-1",
  ozupak: "Özüpak, Y., Alpsalaz, F., Aslan, E.: Air quality forecasting using machine learning: comparative analysis and ensemble strategies for enhanced prediction. Water Air Soil Pollut. 236, 464 (2025). https://doi.org/10.1007/s11270-025-08122-8",
  sidhu: "Sidhu, K.K., Balogun, H., Oseni, K.O.: Predictive modelling of air quality index (AQI) across diverse cities and states of India using machine learning: investigating the influence of Punjab's stubble burning on AQI variability. arXiv preprint arXiv:2404.08702 (2024)",
  rao: "Rao, R.S., Kalabarige, L.R., Alankar, B., Sahu, A.K.: Multimodal imputation-based stacked ensemble for prediction and classification of air quality index in Indian cities. Comput. Electr. Eng. 114, 109098 (2024). https://doi.org/10.1016/j.compeleceng.2024.109098",
  lawrence: "Lawrence, S., Bhathmanabhan, S.: Evaluating machine learning models and imputation strategies for air quality index forecasting in urban India. Environ. Monit. Assess. 197, 1303 (2025). https://doi.org/10.1007/s10661-025-14700-4",
  kapoor: "Kapoor, S., Narayanan, A.: Leakage and the reproducibility crisis in machine-learning-based science. Patterns 4(9), 100804 (2023). https://doi.org/10.1016/j.patter.2023.100804",
  hyndman: "Hyndman, R.J., Athanasopoulos, G.: Forecasting: Principles and Practice, 3rd edn. OTexts, Melbourne (2021)",
  mahato: "Mahato, S., Pal, S., Ghosh, K.G.: Effect of lockdown amid COVID-19 pandemic on air quality of the megacity Delhi, India. Sci. Total Environ. 730, 139086 (2020). https://doi.org/10.1016/j.scitotenv.2020.139086",
  breiman: "Breiman, L.: Random forests. Mach. Learn. 45(1), 5-32 (2001). https://doi.org/10.1023/A:1010933404324",
  cortes: "Cortes, C., Vapnik, V.: Support-vector networks. Mach. Learn. 20(3), 273-297 (1995). https://doi.org/10.1007/BF00994018",
  xgb: "Chen, T., Guestrin, C.: XGBoost: a scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785-794. ACM (2016). https://doi.org/10.1145/2939672.2939785",
  sklearn: "Pedregosa, F., et al.: Scikit-learn: machine learning in Python. J. Mach. Learn. Res. 12, 2825-2830 (2011)",
  pytorch: "Paszke, A., et al.: PyTorch: an imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems, vol. 32, pp. 8024-8035 (2019)",
  adam: "Kingma, D.P., Ba, J.: Adam: a method for stochastic optimization. In: Proceedings of the 3rd International Conference on Learning Representations (ICLR) (2015)",
  demsar: "Demšar, J.: Statistical comparisons of classifiers over multiple data sets. J. Mach. Learn. Res. 7, 1-30 (2006)",
  friedman: "Friedman, M.: The use of ranks to avoid the assumption of normality implicit in the analysis of variance. J. Am. Stat. Assoc. 32(200), 675-701 (1937). https://doi.org/10.1080/01621459.1937.10503522",
  wilcoxon: "Wilcoxon, F.: Individual comparisons by ranking methods. Biometrics Bull. 1(6), 80-83 (1945). https://doi.org/10.2307/3001968",
  holm: "Holm, S.: A simple sequentially rejective multiple test procedure. Scand. J. Stat. 6(2), 65-70 (1979)",
  dm: "Diebold, F.X., Mariano, R.S.: Comparing predictive accuracy. J. Bus. Econ. Stat. 13(3), 253-263 (1995). https://doi.org/10.1080/07350015.1995.10524599",
  hln: "Harvey, D., Leybourne, S., Newbold, P.: Testing the equality of prediction mean squared errors. Int. J. Forecast. 13(2), 281-291 (1997). https://doi.org/10.1016/S0169-2070(96)00719-4",
  shap: "Lundberg, S.M., Lee, S.-I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems, vol. 30, pp. 4765-4774 (2017)",
  treeshap: "Lundberg, S.M., et al.: From local explanations to global understanding with explainable AI for trees. Nat. Mach. Intell. 2(1), 56-67 (2020). https://doi.org/10.1038/s42256-019-0138-9",
};

const order = [];
function cite(...keys) {
  const nums = keys.map((k) => {
    if (!REFS[k]) throw new Error(`unknown reference ${k}`);
    if (!order.includes(k)) order.push(k);
    return order.indexOf(k) + 1;
  });
  return `[${nums.join("], [")}]`;
}
const bibliography = (style = "ieee") => {
  const src = style === "springer" ? LNCS : REFS;
  return order.map((k) => { if (!src[k]) throw new Error(`no ${style} entry for ${k}`); return src[k]; });
};
const unused = () => Object.keys(REFS).filter((k) => !order.includes(k));

module.exports = { cite, bibliography, unused };
