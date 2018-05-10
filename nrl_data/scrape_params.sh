# Quoted species are from NRL_data2.jl
# Further ones entered by hand.
for species in  "ag_par" "al_par" "al_par_t2g_eq_eg" "au_par" "ba_par"\
    "ca_par" "ca_par_315" "co_par" "cr_par" "cu_par" "fe_para_par"\
    "fe_ferro_par" "ga_par" "hf_par" "in_par" "ir_par" "mg_par"\
    "mo_par" "nb_par" "ni_par" "os_par" "pb_par" "pt_par" "re_par"\
    "rh_par" "ru_par" "sc_par" "sr_par" "ta_par" "tc_par" "Ti_par"\
    "ti_01_par" "v_par" "w_par" "y_par" "zr_par"\
    c_par c_par.105\
    si_par.125 si_par.spd si_par\
    ti_par ti_gga_par ti_par_01\
    mn_par\
    fe_par\
    co_para_par co_ferro_par\
    cu_par_99\
    ge_par.sp.125 ge_par.spd.125\
    pd_par.105 pd_par\
    sn_case1_par sn_case2_par
#Nb: ti_01_par above seems incorrect?
do
 wget "https://web.archive.org/web/20121003160812/http://cst-www.nrl.navy.mil/bind/${species}"
done

