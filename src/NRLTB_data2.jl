
function NRLHamiltonian2(s; orbitals=default_orbitals(s), cutoff=:forceshift)
   s = string(s)
   orbitals = string(orbitals)
   if s == "C" && orbitals == "sp"
      H = C_sp
   elseif s == "Si" && orbitals == "sp"
      H = Si_sp
   elseif s == "Si" && orbitals == "spd"
      H = Si_spd
   elseif s == "Ge" && orbitals == "sp"
      H = Ge_sp
   # elseif s == "Ge" && orbitals == "spd"
   #   H = Ge_spd
   else
      cd("/Users/hjchen/packages/TightBinding.jl/nrl_data")
      fname = read_nrl_data(s)
      M = readdlm(fname)
      H =  NRLHamiltonian{9, Function}(9, 10,	   # norbital, nbond
                           M[3,1], M[3,2], cutoff_NRL,			# Rc, lc
                           M[7,1],		# λ
                           [M[8,1],   M[12,1],  M[12,1],  M[12,1],
                            M[16,1],  M[16,1],  M[16,1],  M[16,1], M[16,1]],        #a
                           [M[9,1],   M[13,1],  M[13,1],  M[13,1],
                            M[17,1],  M[17,1],  M[17,1],  M[17,1], M[17,1]],       	#b
                           [M[10,1],  M[14,1],  M[14,1],  M[14,1],
                            M[18,1],  M[18,1],  M[18,1],  M[18,1], M[18,1]],  	   #c
                           [M[11,1],  M[15,1],  M[15,1],  M[15,1],
                            M[19,1],  M[19,1],  M[19,1],  M[19,1], M[19,1]],       	#d

                           [M[24,1],   M[28,1],  M[32,1],  M[36,1],  M[40,1],
                            M[44,1],   M[48,1],  M[52,1],  M[56,1],  M[60,1]],		#e
                           [M[25,1],   M[29,1],  M[33,1],  M[37,1],  M[41,1],
                            M[45,1],   M[49,1],  M[53,1],  M[57,1],  M[61,1]],      #f
                           [M[26,1],   M[30,1],  M[34,1],  M[38,1],  M[42,1],
                            M[46,1],   M[50,1],  M[54,1],  M[58,1],  M[62,1]],      #g
                           [M[27,1],   M[31,1],  M[35,1],  M[39,1],  M[43,1],
                            M[47,1],   M[51,1],  M[55,1],  M[59,1],  M[63,1]],      #h

                           [M[64,1],   M[68,1],   M[72,1],   M[76,1],   M[80,1],
                            M[84,1],   M[88,1],   M[92,1],   M[96,1],   M[100,1]],  #p
                           [M[65,1],   M[69,1],   M[73,1],   M[77,1],   M[81,1],
                            M[85,1],   M[89,1],   M[93,1],   M[97,1],   M[101,1]],  #q
                           [M[66,1],   M[70,1],   M[74,1],   M[78,1],   M[82,1],
                            M[86,1],   M[90,1],   M[94,1],   M[98,1],   M[102,1]],  #r
                           [0.0,   0.0,   0.0,   0.0,   0.0,
                            0.0,   0.0,   0.0,   0.0,   0.0],     						#s
                           [M[67,1],   M[71,1],   M[75,1],   M[79,1],   M[83,1],
                            M[87,1],   M[91,1],   M[95,1],   M[99,1],   M[103,1]],  #t
                           )
   #   error("unkown species / orbitals combination in `NRLParams`")
   end

   if cutoff == :original
      H.fcut = cutoff_NRL_original
   elseif cutoff == :energyshift
      H.fcut = cutoff_NRL_Eshift
   elseif cutoff == :forceshift
      H.fcut = cutoff_NRL_Fshift
   else
      error("unknown cut-off type")
   end
   return H
end


######################## READ NRL-TB DATAS FROM TEXT ###############################

function read_nrl_data(s)
   s = string(s)
   if s == "Ag"
      fname = "ag_par"
   elseif s == "Al"
      fname = "al_par"
   elseif s == "Al_t2g"
      fname = "al_par_t2g_eq_eg"
   elseif s == "Au"
      fname = "au_par"
   elseif s == "Ba"
      fname = "ba_par"
   elseif s == "Ca"
      fname = "ca_par"
   elseif s == "Ca_315"
      fname = "ca_par_315"
   elseif s == "Co"
      fname = "co_par"
   elseif s == "Cr"
      fname = "cr_par"
   elseif s == "Cu"
      fname = "cu_par"
   elseif s == "Fe"
      fname = "fe_para_par"
   elseif s == "Fe_spin"
      fname = "fe_ferro_par"
   elseif s == "Ga"
      fname = "ga_par"
   elseif s == "Hf"
      fname = "hf_par"
   elseif s == "In"
      fname = "in_par"
   elseif s == "Ir"
      fname = "ir_par"
   elseif s == "Mg"
      fname = "mg_par"
   elseif s == "Mo"
      fname = "mo_par"
   elseif s == "Nb"
      fname = "nb_par"
   elseif s == "Ni"
      fname = "ni_par"
   elseif s == "Os"
      fname = "os_par"
   elseif s == "Pb"
      fname = "pb_par"
   elseif s == "Pt"
      fname = "pt_par"
   elseif s == "Re"
      fname = "re_par"
   elseif s == "Rh"
      fname = "rh_par"
   elseif s == "Ru"
      fname = "ru_par"
   elseif s == "Sc"
      fname = "sc_par"
   elseif s == "Sr"
      fname = "sr_par"
   elseif s == "Ta"
      fname = "ta_par"
   elseif s == "Tc"
      fname = "tc_par"
   elseif s == "Ti"
      fname = "Ti_par"
   elseif s == "Ti_01"
      fname = "ti_01_par"
   elseif s == "V"
      fname = "v_par"
   elseif s == "W"
      fname = "w_par"
   elseif s == "Y"
      fname = "y_par"
   elseif s == "Zr"
      fname = "zr_par"
   else
      error("unknown species in `NRLParams`")
   end
   return fname
end
