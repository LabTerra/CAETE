! ATTENTION: this is the simplified version of the file, with less checks and more focused on the allocation itself.
! The complete version of the file, with all the cheks of erros and carbon balance, is run_carbon_allocation_offline_complete.f90
program run_carbon_allocation_offline
   
   use, intrinsic :: iso_fortran_env, only: real64
   use carbon_allocation_offline_kernel

   implicit none

   ! Length of each independent offline simulation.
   integer, parameter :: n_days = 365 * 10

   ! Number of NPP scenarios. Each scenario has its own target mean,
   ! variability, persistence, seasonal amplitude, and allowed bounds.
   ! Increase this value and add entries to the arrays below to run more cases.
   integer, parameter :: n_npp_cases = 1

   ! Target mean of the annualized NPP rate used during daily allocation.
   ! The generated series is shifted after applying the bounds so that its
   ! realized long-term mean is approximately equal to this value.
   real(real64), parameter :: target_mean_npp_values(n_npp_cases) = [ &
      3.5_real64 &
   ]

   ! Standard deviation of the latent autocorrelated anomaly before NPP bounds
   ! are applied. Larger values produce stronger day-to-day NPP variability.
   real(real64), parameter :: anomaly_sd_values(n_npp_cases) = [ &
      2.5_real64 &
   ]

   ! Lag-one persistence of the anomaly. A value of zero gives independent
   ! daily anomalies, whereas values close to one produce multi-day periods
   ! of persistently high or low NPP. A value of 0.90 corresponds to an
   ! approximate anomaly memory of ten days.
   real(real64), parameter :: persistence_values(n_npp_cases) = [ &
      0.90_real64 &
   ]

   ! Amplitude of an optional annual sinusoidal cycle in the annualized NPP
   ! rate. Set this value to zero to use only autocorrelated anomalies.
   real(real64), parameter :: seasonal_amplitude_values(n_npp_cases) = [ &
      0.0_real64 &
   ]

   ! Lower and upper limits imposed on the annualized daily NPP rate.
   ! Negative values are allowed so that respiration can exceed photosynthesis
   ! on unfavorable days, but NPP cannot exceed the prescribed upper limit.
   real(real64), parameter :: minimum_npp_values(n_npp_cases) = [ &
      -3.5_real64 &
   ]

   real(real64), parameter :: maximum_npp_values(n_npp_cases) = [ &
      5.0_real64 &
   ]

   ! Reproducible random seed used to generate the Gaussian innovations.
   ! Keeping the seed fixed produces the same NPP trajectories at every run.
   integer, parameter :: random_seed_value = 20260712

   ! Initial amount of labile carbon available before the first timestep.
   real(real64), parameter :: initial_storage = 0.5_real64

   type(Parameters) :: params
   type(ControlsParam) :: controls
   type(PlantCarbonState) :: state
   type(AllocationOutput) :: result

   real(real64), allocatable :: gaussian_innovations(:)
   real(real64), allocatable :: npp_series(:,:)

   real(real64) :: carbon_storage
   real(real64) :: current_npp_rate
   real(real64) :: living_carbon_final
   real(real64) :: stem_carbon_final
   real(real64) :: structural_carbon_final
   real(real64) :: total_plant_carbon_final

   real(real64) :: realized_mean_npp
   real(real64) :: realized_minimum_npp
   real(real64) :: realized_maximum_npp
   real(real64) :: negative_day_fraction
   real(real64) :: lag_one_autocorrelation

   integer :: i_npp
   integer :: day
   integer :: daily_output_unit
   integer :: final_output_unit

   call initialize_parameters(params)
   call initialize_controls(controls)

   allocate(gaussian_innovations(n_days))
   allocate(npp_series(n_days, n_npp_cases))

   ! Generate one innovation sequence and use it in every scenario. This keeps
   ! the timing of random favorable and unfavorable events comparable among
   ! scenarios when only the NPP parameters are changed.
   call set_reproducible_random_seed(random_seed_value)
   call generate_gaussian_innovations(gaussian_innovations)

   do i_npp = 1, n_npp_cases
      call generate_autocorrelated_npp_series( &
         gaussian_innovations, &
         target_mean_npp_values(i_npp), &
         anomaly_sd_values(i_npp), &
         persistence_values(i_npp), &
         seasonal_amplitude_values(i_npp), &
         minimum_npp_values(i_npp), &
         maximum_npp_values(i_npp), &
         npp_series(:, i_npp))
   end do

   open(newunit=daily_output_unit, &
        file="carbon_allocation_daily.csv", &
        status="replace", action="write")

   open(newunit=final_output_unit, &
        file="carbon_allocation_final.csv", &
        status="replace", action="write")

   call write_daily_header(daily_output_unit)
   call write_final_header(final_output_unit)

   do i_npp = 1, n_npp_cases

      call initialize_state(params, state)
      carbon_storage = initial_storage

      do day = 1, n_days

         ! The module expects an annualized NPP rate. It internally multiplies
         ! this value by dt_years to obtain the carbon input for the current day.
         current_npp_rate = npp_series(day, i_npp)

         call allocate_gradual_with_storage( &
            state, params, controls, current_npp_rate, carbon_storage, result)

         state%leaf_mass = result%leaf_mass_new
         state%root_mass = result%root_mass_new
         state%sapwood_mass = result%sapwood_mass_new
         state%heartwood_mass = result%heartwood_mass_new
         state%height = result%height_new

         call write_daily_row( &
            daily_output_unit, i_npp, &
            target_mean_npp_values(i_npp), current_npp_rate, day, &
            state, carbon_storage, result)

      end do

      living_carbon_final = state%leaf_mass + &
                            state%root_mass + &
                            state%sapwood_mass

      stem_carbon_final = state%sapwood_mass + state%heartwood_mass

      structural_carbon_final = living_carbon_final + state%heartwood_mass

      total_plant_carbon_final = structural_carbon_final + carbon_storage

      call calculate_npp_statistics( &
         npp_series(:, i_npp), &
         realized_mean_npp, &
         realized_minimum_npp, &
         realized_maximum_npp, &
         negative_day_fraction, &
         lag_one_autocorrelation)

      call write_final_row( &
         final_output_unit, i_npp, &
         target_mean_npp_values(i_npp), &
         anomaly_sd_values(i_npp), &
         persistence_values(i_npp), &
         seasonal_amplitude_values(i_npp), &
         minimum_npp_values(i_npp), &
         maximum_npp_values(i_npp), &
         realized_mean_npp, realized_minimum_npp, realized_maximum_npp, &
         negative_day_fraction, lag_one_autocorrelation, &
         state, carbon_storage, &
         living_carbon_final, stem_carbon_final, structural_carbon_final, &
         total_plant_carbon_final, result)

      write(*,'(/,a,i0)') "NPP case: ", i_npp
      write(*,'(a,f12.6)') "Target mean NPP:          ", &
         target_mean_npp_values(i_npp)
      write(*,'(a,f12.6)') "Realized mean NPP:        ", realized_mean_npp
      write(*,'(a,f12.6)') "Realized minimum NPP:     ", realized_minimum_npp
      write(*,'(a,f12.6)') "Realized maximum NPP:     ", realized_maximum_npp
      write(*,'(a,f12.6)') "Fraction of negative days:", negative_day_fraction
      write(*,'(a,f12.6)') "Lag-one autocorrelation:  ", lag_one_autocorrelation
      write(*,'(a,f12.6)') "Final leaf carbon:        ", state%leaf_mass
      write(*,'(a,f12.6)') "Final fine-root carbon:   ", state%root_mass
      write(*,'(a,f12.6)') "Final sapwood carbon:     ", state%sapwood_mass
      write(*,'(a,f12.6)') "Final heartwood carbon:   ", state%heartwood_mass
      write(*,'(a,f12.6)') "Final storage carbon:     ", carbon_storage
      write(*,'(a,f12.6)') "Final total plant carbon: ", total_plant_carbon_final

   end do

   close(daily_output_unit)
   close(final_output_unit)

   deallocate(gaussian_innovations)
   deallocate(npp_series)

   write(*,'(/,a)') "All autocorrelated NPP simulations completed."
   write(*,'(a)') &
      "Daily trajectories: carbon_allocation_daily.csv"
   write(*,'(a)') &
      "Final results: carbon_allocation_final.csv"

contains

   subroutine initialize_parameters(params)

      type(Parameters), intent(out) :: params

      ! Fixed parameter values used in every offline simulation.
      params%sla = 12.0_real64
      params%latosa = 8000.0_real64
      params%wood_density = 250.0_real64
      params%leaf_to_root_ratio = 1.0_real64
      params%allom2 = 40.0_real64
      params%allom3 = 0.5_real64

   end subroutine initialize_parameters


   subroutine initialize_controls(controls)

      type(ControlsParam), intent(out) :: controls

      controls%dt_years = 1.0_real64 / 365.0_real64
      controls%allometric_adjustment_days = 365.0_real64
      controls%max_allocation_fraction = 0.005_real64
      controls%leaf_background_timescale_years = 3.0_real64
      controls%root_background_timescale_years = 3.0_real64
      controls%sapwood_background_timescale_years = 15.0_real64

   end subroutine initialize_controls


   subroutine initialize_state(params, state)

      type(Parameters), intent(in) :: params
      type(PlantCarbonState), intent(out) :: state

      state%leaf_mass = 1.0_real64
      state%root_mass = state%leaf_mass / params%leaf_to_root_ratio
      state%heartwood_mass = 20.0_real64
      state%sapwood_mass = solve_sapwood_for_pipe_balance( &
         params, state%leaf_mass, state%heartwood_mass)
      state%height = height_from_total_stem_carbon( &
         params, state%sapwood_mass + state%heartwood_mass)

   end subroutine initialize_state


   subroutine set_reproducible_random_seed(base_seed)

      integer, intent(in) :: base_seed
      integer, allocatable :: seed(:)
      integer :: seed_size
      integer :: i

      call random_seed(size=seed_size)
      allocate(seed(seed_size))

      do i = 1, seed_size
         seed(i) = base_seed + 104729 * (i - 1)
      end do

      call random_seed(put=seed)
      deallocate(seed)

   end subroutine set_reproducible_random_seed


   subroutine generate_gaussian_innovations(innovations)

      real(real64), intent(out) :: innovations(:)

      real(real64) :: uniform_1
      real(real64) :: uniform_2
      real(real64) :: radius
      real(real64) :: angle
      integer :: i

      ! Box-Muller transformation from independent uniform random numbers to
      ! independent standard Gaussian innovations with mean zero and variance one.
      i = 1

      do while (i <= size(innovations))

         call random_number(uniform_1)
         call random_number(uniform_2)

         uniform_1 = max(uniform_1, tiny(1.0_real64))

         radius = sqrt(-2.0_real64 * log(uniform_1))
         angle = 2.0_real64 * acos(-1.0_real64) * uniform_2

         innovations(i) = radius * cos(angle)

         if (i + 1 <= size(innovations)) then
            innovations(i + 1) = radius * sin(angle)
         end if

         i = i + 2

      end do

   end subroutine generate_gaussian_innovations


   subroutine generate_autocorrelated_npp_series( &
      innovations, target_mean, anomaly_sd, persistence, seasonal_amplitude, &
      minimum_npp, maximum_npp, npp_series)

      real(real64), intent(in) :: innovations(:)
      real(real64), intent(in) :: target_mean
      real(real64), intent(in) :: anomaly_sd
      real(real64), intent(in) :: persistence
      real(real64), intent(in) :: seasonal_amplitude
      real(real64), intent(in) :: minimum_npp
      real(real64), intent(in) :: maximum_npp
      real(real64), intent(out) :: npp_series(:)

      real(real64), allocatable :: raw_pattern(:)
      real(real64) :: anomaly
      real(real64) :: innovation_scale
      real(real64) :: seasonal_phase
      integer :: day

      allocate(raw_pattern(size(npp_series)))

      ! The factor sqrt(1 - persistence^2) makes anomaly_sd the stationary
      ! standard deviation of the AR(1) anomaly before clipping.
      innovation_scale = anomaly_sd * &
                         sqrt(max(0.0_real64, &
                                  1.0_real64 - persistence**2))

      anomaly = anomaly_sd * innovations(1)

      do day = 1, size(npp_series)

         if (day > 1) then
            anomaly = persistence * anomaly + &
                      innovation_scale * innovations(day)
         end if

         seasonal_phase = 2.0_real64 * acos(-1.0_real64) * &
            real(mod(day - 1, 365), real64) / 365.0_real64

         raw_pattern(day) = anomaly + &
            seasonal_amplitude * sin(seasonal_phase)

      end do

      ! Add one constant offset and apply the prescribed bounds. The offset is
      ! found numerically so that the bounded series retains the target mean.
      call shift_and_clip_to_target_mean( &
         raw_pattern, target_mean, minimum_npp, maximum_npp, npp_series)

      deallocate(raw_pattern)

   end subroutine generate_autocorrelated_npp_series


   subroutine shift_and_clip_to_target_mean( &
      raw_pattern, target_mean, minimum_npp, maximum_npp, bounded_series)

      real(real64), intent(in) :: raw_pattern(:)
      real(real64), intent(in) :: target_mean
      real(real64), intent(in) :: minimum_npp
      real(real64), intent(in) :: maximum_npp
      real(real64), intent(out) :: bounded_series(:)

      real(real64) :: lower_shift
      real(real64) :: upper_shift
      real(real64) :: trial_shift
      real(real64) :: trial_mean
      integer :: iteration

      ! The mean of the clipped series is monotonic in the added offset, so a
      ! bisection search can recover the requested bounded-series mean.
      lower_shift = minimum_npp - maxval(raw_pattern) - &
                    abs(maximum_npp - minimum_npp)

      upper_shift = maximum_npp - minval(raw_pattern) + &
                    abs(maximum_npp - minimum_npp)

      do iteration = 1, 200

         trial_shift = 0.5_real64 * (lower_shift + upper_shift)

         bounded_series = max( &
            minimum_npp, &
            min(maximum_npp, raw_pattern + trial_shift))

         trial_mean = sum(bounded_series) / real(size(bounded_series), real64)

         if (trial_mean < target_mean) then
            lower_shift = trial_shift
         else
            upper_shift = trial_shift
         end if

      end do

      trial_shift = 0.5_real64 * (lower_shift + upper_shift)

      bounded_series = max( &
         minimum_npp, &
         min(maximum_npp, raw_pattern + trial_shift))

   end subroutine shift_and_clip_to_target_mean


   subroutine calculate_npp_statistics( &
      series, mean_value, minimum_value, maximum_value, &
      negative_fraction, lag_one_correlation)

      real(real64), intent(in) :: series(:)
      real(real64), intent(out) :: mean_value
      real(real64), intent(out) :: minimum_value
      real(real64), intent(out) :: maximum_value
      real(real64), intent(out) :: negative_fraction
      real(real64), intent(out) :: lag_one_correlation

      real(real64) :: mean_previous
      real(real64) :: mean_next
      real(real64) :: covariance
      real(real64) :: variance_previous
      real(real64) :: variance_next

      mean_value = sum(series) / real(size(series), real64)
      minimum_value = minval(series)
      maximum_value = maxval(series)

      negative_fraction = real(count(series < 0.0_real64), real64) / &
                          real(size(series), real64)

      mean_previous = sum(series(1:size(series) - 1)) / &
                      real(size(series) - 1, real64)

      mean_next = sum(series(2:size(series))) / &
                  real(size(series) - 1, real64)

      covariance = sum( &
         (series(1:size(series) - 1) - mean_previous) * &
         (series(2:size(series)) - mean_next))

      variance_previous = sum( &
         (series(1:size(series) - 1) - mean_previous)**2)

      variance_next = sum( &
         (series(2:size(series)) - mean_next)**2)

      lag_one_correlation = covariance / &
         sqrt(variance_previous * variance_next)

   end subroutine calculate_npp_statistics


   function height_from_total_stem_carbon(params, stem_carbon_total) &
      result(height)

      type(Parameters), intent(in) :: params
      real(real64), intent(in) :: stem_carbon_total
      real(real64) :: height
      real(real64) :: exponent
      real(real64) :: height_power

      exponent = 1.0_real64 + 2.0_real64 / params%allom3

      height_power = params%allom2**(2.0_real64 / params%allom3) * &
                     (stem_carbon_total / params%wood_density) / &
                     (acos(-1.0_real64) / 4.0_real64)

      height = height_power**(1.0_real64 / exponent)

   end function height_from_total_stem_carbon


   function solve_sapwood_for_pipe_balance(params, leaf_mass, heartwood_mass) &
      result(sapwood_mass)

      type(Parameters), intent(in) :: params
      real(real64), intent(in) :: leaf_mass
      real(real64), intent(in) :: heartwood_mass
      real(real64) :: sapwood_mass
      real(real64) :: lower_bound
      real(real64) :: upper_bound
      real(real64) :: midpoint
      integer :: iteration

      lower_bound = 1.0e-12_real64
      upper_bound = 1000.0_real64

      do iteration = 1, 200

         midpoint = 0.5_real64 * (lower_bound + upper_bound)

         if (pipe_balance_residual( &
             params, leaf_mass, heartwood_mass, midpoint) > 0.0_real64) then
            lower_bound = midpoint
         else
            upper_bound = midpoint
         end if

      end do

      sapwood_mass = 0.5_real64 * (lower_bound + upper_bound)

   end function solve_sapwood_for_pipe_balance


   function pipe_balance_residual( &
      params, leaf_mass, heartwood_mass, sapwood_mass) result(residual)

      type(Parameters), intent(in) :: params
      real(real64), intent(in) :: leaf_mass
      real(real64), intent(in) :: heartwood_mass
      real(real64), intent(in) :: sapwood_mass
      real(real64) :: residual
      real(real64) :: height
      real(real64) :: leaf_area
      real(real64) :: sapwood_area

      height = height_from_total_stem_carbon( &
         params, sapwood_mass + heartwood_mass)

      leaf_area = leaf_mass * params%sla
      sapwood_area = sapwood_mass / (params%wood_density * height)

      residual = leaf_area - params%latosa * sapwood_area

   end function pipe_balance_residual


   subroutine write_daily_header(output_unit)

      integer, intent(in) :: output_unit

      write(output_unit,'(a)') &
         "npp_case,target_mean_npp,npp_rate,day,year,npp_daily," // &
         "leaf,root,sapwood,heartwood,height,storage," // &
         "total_demand,carbon_allocated," // &
         "leaf_demand,root_demand,sapwood_demand," // &
         "delta_leaf,delta_root,delta_sapwood," // &
         "unmet_storage_deficit,unpaid_carbon_deficit," // &
         "leaf_turnover,root_turnover,sapwood_turnover," // &
         "storage_turnover,heartwood_turnover," // &
         "leaf_root_residual,pipe_model_residual"

   end subroutine write_daily_header


   subroutine write_daily_row( &
      output_unit, npp_case, target_mean_npp, current_npp_rate, day, &
      state, carbon_storage, result)

      integer, intent(in) :: output_unit
      integer, intent(in) :: npp_case
      integer, intent(in) :: day
      real(real64), intent(in) :: target_mean_npp
      real(real64), intent(in) :: current_npp_rate
      type(PlantCarbonState), intent(in) :: state
      real(real64), intent(in) :: carbon_storage
      type(AllocationOutput), intent(in) :: result

      write(output_unit,'(*(g0,:,","))') &
         npp_case, &
         target_mean_npp, &
         current_npp_rate, &
         day, &
         real(day, real64) / 365.0_real64, &
         result%npp_daily, &
         state%leaf_mass, &
         state%root_mass, &
         state%sapwood_mass, &
         state%heartwood_mass, &
         state%height, &
         carbon_storage, &
         result%total_demand_daily, &
         result%carbon_to_allocate, &
         result%leaf_demand_daily, &
         result%root_demand_daily, &
         result%sapwood_demand_daily, &
         result%delta_leaf, &
         result%delta_root, &
         result%delta_sapwood, &
         result%unmet_storage_deficit, &
         result%unpaid_carbon_deficit, &
         result%leaf_turnover_loss, &
         result%root_turnover_loss, &
         result%sapwood_turnover_loss, &
         result%storage_turnover_loss, &
         result%heartwood_turnover_loss, &
         result%leaf_root_residual, &
         result%pipe_model_residual

   end subroutine write_daily_row


   subroutine write_final_header(output_unit)

      integer, intent(in) :: output_unit

      write(output_unit,'(a)') &
         "npp_case,target_mean_npp,anomaly_sd,persistence," // &
         "seasonal_amplitude,minimum_npp_bound,maximum_npp_bound," // &
         "realized_mean_npp,realized_minimum_npp,realized_maximum_npp," // &
         "negative_day_fraction,lag_one_autocorrelation,simulation_days," // &
         "final_leaf,final_root,final_sapwood,final_heartwood," // &
         "final_storage,final_living_carbon,final_stem_carbon," // &
         "final_structural_carbon,final_total_plant_carbon," // &
         "final_height,final_leaf_root_residual,final_pipe_model_residual"

   end subroutine write_final_header


   subroutine write_final_row( &
      output_unit, npp_case, target_mean_npp, anomaly_sd, persistence, &
      seasonal_amplitude, minimum_npp_bound, maximum_npp_bound, &
      realized_mean_npp, realized_minimum_npp, realized_maximum_npp, &
      negative_day_fraction, lag_one_autocorrelation, &
      state, carbon_storage, living_carbon, stem_carbon, &
      structural_carbon, total_plant_carbon, result)

      integer, intent(in) :: output_unit
      integer, intent(in) :: npp_case
      real(real64), intent(in) :: target_mean_npp
      real(real64), intent(in) :: anomaly_sd
      real(real64), intent(in) :: persistence
      real(real64), intent(in) :: seasonal_amplitude
      real(real64), intent(in) :: minimum_npp_bound
      real(real64), intent(in) :: maximum_npp_bound
      real(real64), intent(in) :: realized_mean_npp
      real(real64), intent(in) :: realized_minimum_npp
      real(real64), intent(in) :: realized_maximum_npp
      real(real64), intent(in) :: negative_day_fraction
      real(real64), intent(in) :: lag_one_autocorrelation
      type(PlantCarbonState), intent(in) :: state
      real(real64), intent(in) :: carbon_storage
      real(real64), intent(in) :: living_carbon
      real(real64), intent(in) :: stem_carbon
      real(real64), intent(in) :: structural_carbon
      real(real64), intent(in) :: total_plant_carbon
      type(AllocationOutput), intent(in) :: result

      write(output_unit,'(*(g0,:,","))') &
         npp_case, &
         target_mean_npp, &
         anomaly_sd, &
         persistence, &
         seasonal_amplitude, &
         minimum_npp_bound, &
         maximum_npp_bound, &
         realized_mean_npp, &
         realized_minimum_npp, &
         realized_maximum_npp, &
         negative_day_fraction, &
         lag_one_autocorrelation, &
         n_days, &
         state%leaf_mass, &
         state%root_mass, &
         state%sapwood_mass, &
         state%heartwood_mass, &
         carbon_storage, &
         living_carbon, &
         stem_carbon, &
         structural_carbon, &
         total_plant_carbon, &
         state%height, &
         result%leaf_root_residual, &
         result%pipe_model_residual

   end subroutine write_final_row

end program run_carbon_allocation_offline
