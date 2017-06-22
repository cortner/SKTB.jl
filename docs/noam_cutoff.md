On Apr 27, 2017, at 6:04 AM, Ortner, Christoph <C.Ortner@warwick.ac.uk> wrote:

Hi Noam,

According to the attached paper, Eq. (16), in NRLTB the cut-off multiplier is this:

cutoff_NRL(r, Rc, lc, Mc = 5.0) =
   (1.0 ./ (1.0 + exp( (r-Rc) / lc + Mc )) .* (r .<= Rc)

See also attached screen-shot. This is not even continuous. In our implementation, we energy-shift it to make it continuous and we play with Mc >> Mc = 10 to be precise, then we get at least within O(1e-6) of a continuously differentiable function. But with the original parameter Mc = 5, and without energy-shift it is awful.

1) Am I missing something?

2) Is this also the cut-off mechanism used in the NRLTB implementation in QUIP? If not, then what do you do instead? (I want to verify my code against yours)

3) If I modify the cut-off, e.g. by force-shifting (or even curvature shift), and playing with the Mc parameter, how could I ensure (and demonstrate) that the model remains as accurate as the original?

Thanks,
        Christoph



I apparently use the routines below.  I think it’s just multiplying the default NRL-TB cutoff function by another function that is 1 for up to the inner cutoff, then switches to a cos(R) form to go exactly to zero at the outer cutoff.

								Noam


function NRLTB_cutoff_function(this, r, ti, tj)
  type(TBModel_NRL_TB), intent(in) :: this
  real(dp), intent(in) :: r
  integer, intent(in) :: ti, tj
  real(dp) :: NRLTB_cutoff_function

  double precision screen_R0
  double precision cutoff_smooth
  double precision expv

  if (r .gt. 10.0_dp**(-4)) then

    screen_R0 = this%r_cut(ti,tj) - 5.0D0*abs(this%screen_l(ti,tj))

    expv = exp((r-screen_R0)/abs(this%screen_l(ti,tj)))

    cutoff_smooth = NRLTB_cutoff_func_smooth(this, r, ti, tj)
    NRLTB_cutoff_function = ( 1.0_dp/(1.0_dp+expv) ) * cutoff_smooth
  else
    NRLTB_cutoff_function = 0.0_dp
  end if

end function NRLTB_cutoff_function

function NRLTB_cutoff_func_smooth(this, r, ti, tj)
  type(TBModel_NRL_TB), intent(in) :: this
  real(dp), intent(in) :: r
  integer, intent(in) :: ti, tj
  real(dp) :: NRLTB_cutoff_func_smooth

  double precision R_MIN, R_MAX

  double precision PI
  parameter (PI = 3.14159265358979323846264338327950288_dp)


  if (this%screen_l(ti,tj) .lt. 0.0_dp) then
    NRLTB_cutoff_func_smooth = 1.0_dp
    return
  endif

  R_MAX = this%r_cut(ti,tj)
  R_MIN = R_MAX - abs(this%screen_l(ti,tj))

  if (r .lt. R_MIN) then
    NRLTB_cutoff_func_smooth = 1.0_dp
  else if (r .gt. R_MAX) then
    NRLTB_cutoff_func_smooth = 0.0_dp
  else
    NRLTB_cutoff_func_smooth = 1.0_dp - (1.0_dp - cos( (r-R_MIN)*PI / (R_MAX-R_MIN) ))/2.0_dp
  end if

end function NRLTB_cutoff_func_smooth



____________
||
|U.S. NAVAL|
|_RESEARCH_|
LABORATORY
Noam Bernstein, Ph.D.
Center for Materials Physics and Technology
U.S. Naval Research Laboratory
T +1 202 404 8628  F +1 202 404 7546
https://www.nrl.navy.mil
