# Spiral 3DREAM

#%%

import numpy as np
import os
import datetime

from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_adc import make_adc
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_delay import make_delay
from pypulseq.opts import Opts
from pypulseq.calc_duration import calc_duration

import ismrmrd
import spiraltraj

import pulseq_helper as ph
from prot import create_hdr
from gre_3Drefscan import gre_3Drefscan
from gre_3DrefscanLinear import gre_3DrefscanLinear
import phase_enc_helper as pe

#%% Parameters 
"""
PyPulseq units (SI): 
time:       [s] (not [ms] as in documentation)
spatial:    [m]
gradients:  [Hz/m] (gamma*T/m)
grad area:  [1/m]
flip angle: [rad]

Some units get converted below, others have to stay in non-SI units as spiral calculation needs different units

Add new parameters also to the protocol parameter h5 file at the end of this script.
Use custom version of Pypulseq: https://github.com/mavel101/pypulseq (branch dev_mv)
"""

# General
seq_name        = "3Dream_b0corr_2p5mm_1shot"      # sequence/protocol filename
meas_date       = None
plot            = False         # plot sequence (refscans & noisescans will be deactivated)

# Sequence - Contrast and Geometry
fov             = 200           # field of view [mm]
TR              = 8.95          # repetition time [ms]
TE_ste          = 0.92          # STE/ STE* echo time [ms]
TE_fid          = 4.99          # FID echo time [ms]
res             = 2.5           # in plane resolution [mm]                   
Nz              = 80            # number of partitions
averages        = 1             # number of averages
shots           = 1             # number of shots (=number of STEAM preparations)
dummy_shots     = 0             # number of dummy shots (should be 0, if shots = 1)

refscan         = 1             # 0: no refscan, 1: Cartesian spiral two phase encoding order, 2: linear phase encoding order
ref_lines       = 50            # number of reference lines (Nx,Ny & Nz)
b0_corr         = True          # Activate B0 correction (acquires 2-echo reference scan)
prepscans       = 0             # number of preparation/dummy scans before spiral sequence
noisescans      = 16            # number of noise scans

# STEAM prep
flip_angle_ste  = 50            # flip angle of the excitation pulses of the STEAM prep-sequence [°]
rf_dur_ste      = 0.4           # RF duration [ms]
time_scheme     = 3             # 1: TS = TE_ste, 2: TS = TE_ste + TE_fid , 3: TS = TE_fid - TE_ste; for all: STE/ STE* first
TM              = 1.82          # mixing time: time span between the 2nd alpha and 1st beta pulse -> small compared to T1 relaxation time [ms]

# RF - Imaging
flip_angle      = 5             # flip angle of excitation pulse [°]
rf_dur          = 0.2           # RF duration [ms]
tbp_exc         = 6             # time bandwidth product excitation pulse
rf_spoiling     = False         # RF spoiling

fatsat          = True          # Fat saturation pulse
fatsat_dur      = 2.1           # duration of fatsat pulse [ms]

# ADC
os_factor       = 2             # oversampling factor (automatic 2x os from Siemens is not applied)

# Gradients
max_slew        = 200           # maximum slewrate [T/m/s] (system limit)
spiral_slew     = 198           # maximum slew rate of spiral gradients - for pre Emph: set lower than max_slew
max_grad        = 55            # maximum gradient amplitude [mT/m] (system limit) - used also for diffusion gradients
max_grad_sp     = 42            # maximum gradient amplitude of spiral gradients - for pre_emph: set lower than max_grad

Nintl           = 12             # spiral interleaves
redfac          = 3             # in-plane acceleration factor, recommended: 3
spiral_os       = 1             # spiral oversampling in center
Rz              = 2             # acceleration factor in kz direction, recommended: 2
caipi_shift     = 2            # CAIPI-shift

# T1 estimation for global filter approach
t1              = 2             # expected T1 value [s] (780ms GUFI, 2s grey matter)

#%% Limits, checks and preparations

# Set System limits
rf_dead_time = 100e-6       # lead time before rf can be applied
rf_ringdown_time = 30e-6    # coil hold time (20e-6) + frequency reset time (10e-6)
system = Opts(max_grad=max_grad, grad_unit='mT/m', max_slew=max_slew, slew_unit='T/m/s', rf_dead_time=rf_dead_time, rf_ringdown_time=rf_ringdown_time)
B0 = 6.983                  # field strength [T]

# time span between the two STEAM preparation pulses [ms]
if time_scheme == 1:            
    TS = TE_ste
    B0map = True
elif time_scheme == 2:
    TS = TE_ste + TE_fid
    B0map = True
elif time_scheme == 3:
    TS = TE_fid - TE_ste
    B0map = False

# convert parameters to Pulseq units
TD = TM + TS            # time span between the 1st alpha and 1st beta pulse [ms]
slice_res = fov / Nz    # partition thickness
rf_dur_ste  *= 1e-3     # [s]
TR          *= 1e-3     # [s]
TE_ste      *= 1e-3     # [s]
TE_fid      *= 1e-3     # [s]
TS          *= 1e-3     # [s]
TD          *= 1e-3     # [s]
TM          *= 1e-3     # [s]
rf_dur      *= 1e-3     # [s]
slice_res   *= 1e-3     # [m]
fatsat_dur  *= 1e-3     # [s]

slices = 1 # 3D acquisition
eff_intl = Nintl//redfac
eff_Nz = Nz//Rz
contrasts = 2 # number of contrasts (DREAM: STE/ STE* and FID)

# fatsat/water excitation parameters
fw_shift = 3.3e-6 # unsigned fat water shift [1/s]
fw_shift_b0 = -1 * int(B0*system.gamma*fw_shift) # fat water shit for specific B0

if eff_intl%1 != 0:
    raise ValueError('Number of interleaves is not multiple of reduction factor')
if eff_Nz%1 != 0:
    raise ValueError('Number of partitions is not multiple of reduction factor Rz')
if eff_Nz%2 != 0 or Nz%2 != 0:
    raise ValueError('Number of partitions and effective partitions should be even integer')
if caipi_shift >= redfac:
    raise ValueError('Choose caipi shift smaller than reduction factor.')
if eff_Nz % shots:
    raise ValueError(f'Effective number of partitions {eff_Nz} must be dividable by number of shots.')
if Nz < ref_lines or (fov/res) < ref_lines:
    raise ValueError(f'Number of reference lines {ref_lines} must be smaller than partitions {Nz} and image size {fov/res}.')

#%% RF pulse and slab/slice selection gradient

# make rf pulse and calculate duration of excitation and rewinding -> binomial pulses for water excitation
# WIP: implement non-selective pulse for sagittal??
flip_angle_bin = flip_angle/2
rf, gz, gz_rew, rf_del = make_sinc_pulse(flip_angle=flip_angle_bin*np.pi/180, system=system, duration=rf_dur, slice_thickness=fov*1e-3,
                            apodization=0.5, time_bw_product=tbp_exc, use='excitation', return_gz=True, return_delay=True)
exc_to_rew = rf_del.delay - rf_dur/2 - rf.delay # time from middle of rf pulse to rewinder, rf_del.delay equals the block length
rew_dur = calc_duration(gz_rew)

amp_gz_rew_bin, ftop_gz_rew_bin, ramp_gz_rew_bin = ph.trap_from_area(2*gz_rew.area, system, slewrate=130)
gz_rew_bin= make_trapezoid(channel='z', system=system, amplitude=amp_gz_rew_bin, flat_time=ftop_gz_rew_bin, rise_time=ramp_gz_rew_bin)

tau_bin = abs(1 / (2*fw_shift_b0)) # [s]
min_tau_bin = calc_duration(rf,gz,rf_del) + calc_duration(gz_rew_bin)
if min_tau_bin > tau_bin:
    min_tau_bin = ph.round_up_to_raster(min_tau_bin, decimals=5)*1e3
    raise ValueError('tau_bin must be at least {} ms'.format(min_tau_bin)) 
delay_tau_bin = round(tau_bin - min_tau_bin,5)
tau_bin_delay = make_delay(d=delay_tau_bin)

# RF spoiling parameters
rf_spoiling_inc = 50 # increment of RF spoiling [°]
rf_phase        = 0 
rf_inc          = 0

# Fat saturation
if fatsat:
    fatsat_bw = 1000 # bandwidth [Hz] (1000 Hz is approx. used by Siemens)
    fatsat_fa = 110 # flip angle [°]
    fatsat_tbp = 2.1 
    fatsat_dur = ph.round_up_to_raster(fatsat_tbp/fatsat_bw, decimals=5)
    rf_fatsat, fatsat_del = make_gauss_pulse(flip_angle=fatsat_fa*np.pi/180, duration=fatsat_dur, bandwidth=fatsat_bw, freq_offset=fw_shift_b0, system=system, return_delay = True)

#%% STEAM preparation

# rf pulses (block pulses)
rf_ste, rf_ste_delay = make_block_pulse(flip_angle=flip_angle_ste*np.pi/180, duration=rf_dur_ste, return_delay=True,system=system, use='excitation')

# Area A1 for k-space separation
c = 3                   # determines the separation in k-space
Nx = int(fov/res+0.5)
delta_k = 1/(fov*1e-3)  #[1/m]
A1 = c * Nz*delta_k/2   #[1/m]

# prephaser Gm1 (in the imaging train)
gm1_area = -A1          #[1/m]

# rephaser Gmiddle (in the imaging train)
gmiddle_area = A1       #[1/m]
amp_gmiddle, ftop_gmiddle, ramp_gmiddle = ph.trap_from_area(gmiddle_area, system)
gmiddle = make_trapezoid(channel='z', system=system, amplitude=amp_gmiddle, flat_time=ftop_gmiddle, rise_time=ramp_gmiddle)

# dephaser Gm2 (in the STEAM prep. seq.)
if (time_scheme == 1 or time_scheme == 2):
    gm2_area = -A1      #[1/m]
elif time_scheme == 3:
    gm2_area = A1       #[1/m]
amp_gm2, ftop_gm2, ramp_gm2 = ph.trap_from_area(gm2_area, system)
gm2 = make_trapezoid(channel='z', system=system, amplitude=amp_gm2, flat_time=ftop_gm2, rise_time=ramp_gm2) # in phase encoding direction

# spoiler (after second rf pulse)
gx_spoil_ste = make_trapezoid(channel='x', area=4 /(res*1e-3), system=system, max_slew=80*system.gamma)
gy_spoil_ste = make_trapezoid(channel='y', area=4 /(res*1e-3), system=system, max_slew=80*system.gamma)
gz_spoil_ste = make_trapezoid(channel='z', area=4 /(slice_res),system=system, max_slew=80*system.gamma)

# Delays
x_ste = calc_duration(rf_ste) - rf_ste.dead_time    # [s]
a_ste = rf_ste.dead_time + x_ste/2                  # [s]
b_ste = rf_ste_delay.delay - a_ste                  # [s]

# take minimum TS rounded up to .1 ms
min_TS = b_ste + calc_duration(gm2) + a_ste         # [s]
if min_TS > TS:
    min_TS = ph.round_up_to_raster(min_TS, decimals=5)*1e3
    raise ValueError('TS must be at least {} ms'.format(min_TS))
delay_TS = round(TS - min_TS,5)
ts_delay = make_delay(d=delay_TS)

# take minimum TM rounded up to .1 ms
min_TM = b_ste + calc_duration(gz_spoil_ste, gx_spoil_ste, gy_spoil_ste) + rf.delay + gz.flat_time/2 # [s]
if min_TM > TM:
    min_TM = ph.round_up_to_raster(min_TM, decimals=5)*1e3
    raise ValueError('TM must be at least {} ms'.format(min_TM)) 
delay_TM = round(TM - min_TM,5)
tm_delay = make_delay(d=delay_TM)

#%% Spiral Readout Gradients

# In the DREAM sequence always Spiraltype 1 = Spiral Out is used

# Parameters spiral trajectory:

# parameter         description               default value
# ---------        -------------              --------------

# nitlv:      number of spiral interleaves        15
# res:        resolution                          1 mm
# fov:        target field of view                192 mm
# max_amp:    maximum gradient amplitude          42 mT/m
# min_rise:   minimum gradient risetime           5 us/(mT/m)
# spiraltype: 1: spiral out                   
#             2: spiral in                        
#             3: double spiral                    x
#             4: ROI
#             5: RIO
# spiral_os:  spiral oversampling in center       1

# Rotation of Spirals
max_rot     = 2*np.pi  

# read in Spirals [T/m]
spiraltype = 1
min_rise_sp = 1/spiral_slew * 1e3
spiral_calc = spiraltraj.calc_traj(nitlv=Nintl, fov=fov, res=res, spiraltype=spiraltype, min_rise=min_rise_sp, max_amp=max_grad_sp, spiral_os=spiral_os) # [mT/m]
spiral_calc = np.asarray(spiral_calc)
spiral_x = 1e-3*spiral_calc[:,0]
spiral_y = 1e-3*spiral_calc[:,1]

N_spiral = len(spiral_x)
readout_dur = N_spiral*system.grad_raster_time # readout duration [s]

# write spiral readout blocks to list
spirals = [{'deph': [None, None], 'spiral': [None, None], 'reph': [None, None]} for k in range(Nintl)]
reph_dur = []
save_sp = np.zeros((Nintl, 2, N_spiral)) # save gradients for FIRE reco
rot_angle = np.linspace(0, max_rot, Nintl, endpoint=False)
for k in range(Nintl):
    # rotate spiral gradients for shot selection
    sp_x, sp_y = ph.rot_grad(spiral_x, spiral_y, rot_angle[k])

    save_sp[k,0,:] = sp_x # for FIRE reco
    save_sp[k,1,:] = sp_y # for FIRE reco

    # unit to [Hz/m], make spiral gradients (conversion into pulseq units)
    sp_x *= system.gamma
    sp_y *= system.gamma
    spiral_delay = 20e-6 # delay spiral relative to begin of ADC
    spirals[k]['spiral'][0] = make_arbitrary_grad(channel='x', waveform=sp_x, delay=spiral_delay, system=system)
    spirals[k]['spiral'][1] = make_arbitrary_grad(channel='y', waveform=sp_y, delay=spiral_delay, system=system)

    # calculate rephaser area
    area_x = sp_x.sum()*system.grad_raster_time
    area_y = sp_y.sum()*system.grad_raster_time

    # calculate rephasers and make gradients
    amp_x, ftop_x, ramp_x = ph.trap_from_area(-area_x, system, slewrate = min(100,max_slew)) # reduce slew rate to 100 T/m/s to avoid stimulation
    amp_y, ftop_y, ramp_y = ph.trap_from_area(-area_y, system, slewrate = min(100,max_slew))
    spirals[k]['reph'][0] = make_trapezoid(channel='x', system=system, amplitude=amp_x, duration=2*ramp_x+ftop_x, rise_time=ramp_x)
    spirals[k]['reph'][1] = make_trapezoid(channel='y', system=system, amplitude=amp_y, duration=2*ramp_y+ftop_y, rise_time=ramp_y)
    reph_dur.append(max(ftop_x+2*ramp_x, ftop_y+2*ramp_y))

# check for acoustic resonances (checks only spirals)
freq_max = ph.check_resonances([spiral_x,spiral_y])

#%% Second phase encoding (in z-direction)

phase_areas, phase_enc_steps = pe.CentricOrder(N=eff_Nz, fov=fov*1e-3, Rpe=Rz)  #[1/m]

# calculate maximum gz prephaser duration
max_area_gzpre = np.max(abs(phase_areas+gz_rew.area+gm1_area))
amp_gz_pre, ftop_gz_pre, ramp_gz_pre = ph.trap_from_area(max_area_gzpre, system, slewrate=min(120,max_slew))
gz_pre = make_trapezoid(channel='z', system=system, amplitude=amp_gz_pre, duration=2*ramp_gz_pre+ftop_gz_pre, rise_time=ramp_gz_pre)
max_dur_gz_pre = calc_duration(gz_pre)

# Calculate minimum TE_ste
min_TE_ste = exc_to_rew + max_dur_gz_pre # [s]
if min_TE_ste > TE_ste:
    min_TE_ste = ph.round_up_to_raster(min_TE_ste, decimals=5)*1e3
    raise ValueError('TE_ste has to be at least {} ms'.format(min_TE_ste)) 
delay_TE_ste = round(TE_ste - min_TE_ste,5)
te_ste_delay = make_delay(d=delay_TE_ste)

#%% Spoilers 

# Calculate maximum duration of the spoiler incl gz rephaser gradient
spoiler_area = 4 / slice_res
max_area_gzreph = np.max(abs(phase_areas+spoiler_area))
amp_gz_spoil, ftop_gz_spoil, ramp_gz_spoil= ph.trap_from_area(max_area_gzreph, system, slewrate=120)
gz_spoil = make_trapezoid(channel='z', system=system, amplitude=amp_gz_spoil, duration=2*ramp_gz_spoil+ftop_gz_spoil, rise_time=ramp_gz_spoil)

#%% ADC

max_grad_sp_cmb = 1e3*np.max(np.sqrt(abs(spiral_x)**2+abs(spiral_y)**2))
dwelltime = 1/(system.gamma*max_grad_sp_cmb*fov*os_factor)*1e6  # ADC dwelltime [s]
dwelltime = ph.trunc_to_raster(dwelltime, decimals=7)           # truncate dwelltime to 100 nanoseconds (scanner limit)
min_dwelltime = 1e-6
if dwelltime < min_dwelltime:
    dwelltime = min_dwelltime
print("ADC dwelltime: {}".format(dwelltime))

num_samples = round((readout_dur+spiral_delay)/dwelltime)
if num_samples%2==1:
    num_samples += 1 # even number of samples
print('Number of ADCs: {}.'.format(num_samples))

if num_samples > 8192:
   raise ValueError("Maximum number of unsegmented ADCs is 8192.")

adc = make_adc(system=system, num_samples=num_samples, dwell=dwelltime)
adc_dur = num_samples * dwelltime
adc_delay = ph.round_up_to_raster(adc_dur+200e-6, decimals=5) # add small delay after readout for ADC frequency reset event and to avoid stimulation by rephaser
adc_delay = make_delay(d=adc_delay)

#%% Calculate minimum TE_fid & minimum TR

# small delay after gmiddle and before 2nd spiral to reduce effect of eddy currents
delay_2spiral = 1e-3 # [s]

# calculate minimum TE_fid
if max(reph_dur) > calc_duration(gmiddle):
    gmiddle = make_trapezoid(channel='z', system=system, duration=max(reph_dur), area=gmiddle_area) # stretch gmiddle, if its shorter than longest rephaser
min_TE_fid = TE_ste + adc_delay.delay + calc_duration(gmiddle) + delay_2spiral # [s]
min_TE_fid = ph.round_up_to_raster(min_TE_fid, decimals=5)
if min_TE_fid > TE_fid:
    raise ValueError('TE_fid has to be at least {} ms'.format(min_TE_fid*1e3))
te_fid_delay = TE_fid - min_TE_fid + delay_2spiral
te_fid_delay = make_delay(d=te_fid_delay)

# calculate minimum TR
if max(reph_dur) > calc_duration(gz_spoil):
    gz_spoil = make_trapezoid(channel='z', system=system, area=gz_spoil.area, duration=max(reph_dur))
max_dur_gzreph = calc_duration(gz_spoil)
min_TR = rf.delay + gz.flat_time/2 + TE_fid + adc_delay.delay + max_dur_gzreph + tau_bin # [s]
if TR < min_TR:
        raise ValueError('Minimum TR is {} ms.'.format(min_TR*1e3))
tr_delay = make_delay(d=TR-min_TR)

#%% Set up protocol for FIRE reco and write header

if meas_date is None:
    meas_date = datetime.date.today().strftime('%Y%m%d')
filename = meas_date + '_' + seq_name

# set up protocol file and create header
mrd_file = "metadata.h5"
if os.path.exists(mrd_file):
    raise ValueError("Protocol name already exists. Choose different name")
prot = ismrmrd.Dataset(mrd_file)
hdr = ismrmrd.xsd.ismrmrdHeader()

t_min_fid = TE_fid + dwelltime/2 # for B0 correction
t_min_ste = TE_ste + TS + dwelltime/2
params_hdr = {"trajtype": "spiral", "fov": fov, "res": res, "slices": slices, "slice_res": slice_res, "npartitions": Nz, "nintl": eff_intl, "avg": averages,
                "dwelltime": dwelltime, "traj_delay": spiral_delay, "ncontrast": contrasts, "redfac1": redfac, "redfac2": Rz, "os_factor": os_factor, "t_min_fid": t_min_fid,
                "t_min_ste": t_min_ste}
create_hdr(hdr, params_hdr)

# append dream array for B1 map calculation (with global filter)
ste_ix = 0 # always STE first
dream = np.array([ste_ix,flip_angle_ste,TR,flip_angle,prepscans,t1,TM,tau_bin])
prot.append_array('dream', dream)

# append array for B0 mapping
if B0map:
    te_diff = TE_ste + TS - TE_fid
    dreamB0 = np.array([te_diff])
    prot.append_array('dreamB0', dreamB0)


#%% Add sequence blocks to sequence & write acquisitions to protocol

# Set up the sequence
seq = Sequence()

# Definitions section in seq file - WIP: What is "Delays" definition doing??
seq.set_definition("Name", filename) # protocol name is saved in Siemens header for FIRE reco
seq.set_definition("FOV", [1e-3*fov, 1e-3*fov, 1e-3*fov]) # for FOV positioning

# Noise scans
if plot:
    print("We do plotting, so noisescans are set to 0.")
    noisescans = 0
noise_samples = 256
noise_adc = make_adc(system=system, num_samples=256, dwell=dwelltime)
noise_delay = make_delay(d=ph.round_up_to_raster(noise_adc.duration+1e-3,decimals=5)) # add some more time to the ADC delay to be safe
for k in range(noisescans):
    seq.add_block(noise_adc, noise_delay)
    acq = ismrmrd.Acquisition()
    acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
    prot.append_acquisition(acq)

# Perform cartesian reference scan: if selected / for accelerated spirals / for long readouts
if refscan == False and (redfac>1 or Rz>1):
    refscan = 1
    print("Accelerated scan: Activate Cartesian reference scan.")
if plot:
    refscan = 0
    print("We do plotting, so refscan is disabled.")

if refscan:
    # reduce resolution for faster scan
    bw_refscan = 1200
    flip_refscan = 5
    dur_refscan  = 2e-3 # refscan pulse duration [ms]
    tbp_refscan  = 23 # time bandwidth product refscan pulse
    if b0_corr:
        TE_refscan = [2.04e-3, 4.08e-3]
    else:
        TE_refscan = None
    if refscan == 1: # Cartesian spiral
        params_ref = {"fov":fov*1e-3, "flip_angle": flip_refscan, "rf_dur": dur_refscan,
         "tbp": tbp_refscan, "readout_bw": bw_refscan, "ref_lines": ref_lines, "TE": TE_refscan}
        gre_3Drefscan(seq, prot=prot, system=system, params=params_ref)
    elif refscan == 2: # linear two phase encoding order
        params_ref = {"fov":fov*1e-3, "flip_angle":flip_refscan, "rf_dur":dur_refscan, 
        "tbp": tbp_refscan, "readout_bw": bw_refscan, "ref_lines": ref_lines, "TE": TE_refscan}
        gre_3DrefscanLinear(seq, prot=prot, system=system, params=params_ref)

""" 
The following code generates the Spiral DREAM sequence.
"""

Nz_per_shot = eff_Nz // shots

for s in range(-dummy_shots, shots):

    # fatsat
    if fatsat:
        seq.add_block(rf_fatsat, fatsat_del)
        seq.add_block(gz_spoil)

    # STEAM preparation
    seq.add_block(rf_ste,rf_ste_delay)
    seq.add_block(gm2)
    seq.add_block(ts_delay)
    seq.add_block(rf_ste,rf_ste_delay)
    seq.add_block(gz_spoil_ste, gx_spoil_ste, gy_spoil_ste)
    seq.add_block(tm_delay)

    # preparation scans without adc
    for k in range(prepscans): 
        if rf_spoiling:
                rf.phase_offset  = rf_phase / 180 * np.pi
                adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
        gz_pre = make_trapezoid(channel='z', area=phase_areas[0] + gz_rew.area + gm1_area, duration=max_dur_gz_pre, system=system)
        seq.add_block(rf,gz,rf_del)
        seq.add_block(gz_rew_bin)
        seq.add_block(tau_bin_delay)
        seq.add_block(rf,gz,rf_del)
        seq.add_block(gz_pre)
        seq.add_block(te_ste_delay)
        seq.add_block(spirals[0]['spiral'][0], spirals[0]['spiral'][1], adc_delay)
        seq.add_block(spirals[0]['reph'][0], spirals[0]['reph'][1], gmiddle)
        seq.add_block(te_fid_delay)
        seq.add_block(spirals[0]['spiral'][0], spirals[0]['spiral'][1], adc_delay)
        gz_reph = make_trapezoid(channel='z', area=-phase_areas[0] + spoiler_area, duration=max_dur_gzreph, system=system)
        seq.add_block(spirals[0]['reph'][0], spirals[0]['reph'][1], gz_reph)
        seq.add_block(tr_delay)

    # imaging scans
    for j in range(Nz_per_shot):
        shot_ix = s if s>=0 else 0
        phs_ix = j*shots + shot_ix

        # phase encoding in z-direction
        gz_pre = make_trapezoid(channel='z', area=phase_areas[phs_ix] + gz_rew.area + gm1_area, duration=max_dur_gz_pre, system=system)
        gz_reph = make_trapezoid(channel='z', area=-phase_areas[phs_ix] + spoiler_area, duration=max_dur_gzreph, system=system)
        
        for n in range(eff_intl):
            nint = (n*redfac + phase_enc_steps[phs_ix]*caipi_shift) % Nintl
            
            if rf_spoiling:
                rf.phase_offset  = rf_phase / 180 * np.pi
                adc.phase_offset = rf_phase / 180 * np.pi

            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
        
            # excitation with prephaser gm1
            seq.add_block(rf,gz,rf_del)
            seq.add_block(gz_rew_bin)
            seq.add_block(tau_bin_delay)
            seq.add_block(rf,gz,rf_del)
            seq.add_block(gz_pre)
        
            # first spiral readout block with spoiler gradient
            seq.add_block(te_ste_delay)
            if s >= 0:
                seq.add_block(spirals[nint]['spiral'][0], spirals[nint]['spiral'][1], adc, adc_delay)
            else:
                seq.add_block(spirals[nint]['spiral'][0], spirals[nint]['spiral'][1], adc_delay)
            seq.add_block(spirals[nint]['reph'][0], spirals[nint]['reph'][1], gmiddle)
            seq.add_block(te_fid_delay)
            
            # second spiral
            if s >= 0:
                seq.add_block(spirals[nint]['spiral'][0], spirals[nint]['spiral'][1], adc, adc_delay)
            else:
                seq.add_block(spirals[nint]['spiral'][0], spirals[nint]['spiral'][1], adc_delay)
            seq.add_block(spirals[nint]['reph'][0], spirals[nint]['reph'][1], gz_reph)

            # delay at end of one TR
            seq.add_block(tr_delay)
        
            # add protocol information
            if s >= 0:
                for contr in range(contrasts):
                    acq = ismrmrd.Acquisition()
                    if (n == eff_intl - 1) and (phs_ix == eff_Nz-1):
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                    acq.idx.kspace_encode_step_1 = n
                    acq.idx.kspace_encode_step_2 = phase_enc_steps[phs_ix]
                    acq.idx.slice = 0
                    acq.idx.contrast = contr
                    acq.idx.average = 0
                    acq.idx.set = shot_ix # save shot number for FID filter
                    # we misuse the trajectory field for the gradient array
                    acq.resize(trajectory_dimensions = save_sp.shape[1], number_of_samples=save_sp.shape[2], active_channels=0)
                    acq.traj[:] = np.swapaxes(save_sp[nint],0,1) # [samples, dims]
                    prot.append_acquisition(acq)
    if s < shots-1:
        seq.add_block(make_delay(d=3)) # delay after shot
        # intl
    # kz
# shots

print("min_TS = {} sec".format(min_TS))
print("min_TM = {} sec".format(min_TM))
print("min_TE_ste = {} sec".format(min_TE_ste))
print("minTE_fid = {} sec".format(min_TE_fid))
print("minTR = {} sec".format(min_TR))   
print("sequence duration = {} sec".format(seq.duration()[0]))

#%% Plot sequence

if plot:
    seq.plot(time_range=(0, 0.015),time_disp = 'ms',save=True)

#%% save sequence

seq.write('sequence.seq')
prot.close()
