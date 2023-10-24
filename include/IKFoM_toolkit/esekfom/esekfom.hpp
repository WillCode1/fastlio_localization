/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP

#include <vector>
#include <cstdlib>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
#include "util.hpp"


namespace esekfom
{
	using namespace Eigen;

	template <typename T>
	struct fastlio_datastruct
	{
		bool valid;
		bool converge;
		Eigen::Matrix<T, Eigen::Dynamic, 1> z;
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> H;
		double R;
	};

	template <typename T>
	struct pointlio_datastruct
	{
		bool valid;
		bool converge;
		T M_Noise;
		Eigen::Matrix<T, Eigen::Dynamic, 1> z;
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
		Eigen::Matrix<T, 6, 1> z_IMU;
		Eigen::Matrix<T, 6, 1> R_IMU;
		bool satu_check[6];
	};

	template <typename state, int process_noise_dof, typename input = state, typename measurement = state, int measurement_noise_dof = 0>
	class esekf
	{
		typedef esekf self;
		enum
		{
			n = state::DOF,
			m = state::DIM,
			l = measurement::DOF
		};

	public:
		typedef typename state::scalar scalar_type;
		typedef Matrix<scalar_type, n, n> cov;
		typedef Matrix<scalar_type, m, n> cov_;
		typedef SparseMatrix<scalar_type> spMt;
		typedef Matrix<scalar_type, n, 1> vectorized_state;
		typedef Matrix<scalar_type, m, 1> flatted_state;
		typedef flatted_state processModel(state &, const input &);
		typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
		typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
		typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;

		// typedef void measurementModel_dyn_share(state &,  fastlio_datastruct<scalar_type> &);
		using measurementModel_dyn_share = std::function<void(state &, fastlio_datastruct<scalar_type> &)>;
		// typedef void measurementModel_dyn_share_modified(state &, pointlio_datastruct<scalar_type> &);
		using measurementModel_dyn_share_modified = std::function<void(state &, pointlio_datastruct<scalar_type> &)>;
		typedef Eigen::Matrix<scalar_type, l, n> measurementMatrix1(state &);
		typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, n> measurementMatrix1_dyn(state &);
		typedef Eigen::Matrix<scalar_type, l, measurement_noise_dof> measurementMatrix2(state &);
		typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &);
		typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
		typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

		esekf(const state &x = state(), const cov &P = cov::Identity())
			: x_(x), P_(P){};

		// for fastlio2
		void init_dyn_share(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in, measurementModel_dyn_share h_dyn_share_in, int maximum_iteration, scalar_type limit_vector[n])
		{
			f = f_in;
			f_x = f_x_in;
			f_w = f_w_in;
			h_dyn_share = h_dyn_share_in;

			maximum_iter = maximum_iteration;
			for (int i = 0; i < n; i++)
			{
				limit[i] = limit_vector[i];
			}

			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
		}

		// iterated error state EKF propogation. for fastlio2
		void predict(double &dt, processnoisecovariance &Q, const input &i_in)
		{
			// 1.imu前向传播
			flatted_state f_ = f(x_, i_in);
			state x_before = x_;
			x_.oplus(f_, dt); // 对应fast-lio2中公式(2)

			// 2.计算前向传播的jacob矩阵F_x和F_w
			// 这里的F_x分成了 对角线F_x1 和 非对角线部分F_x2
			F_x1 = cov::Identity();
			cov_ F_x2 = f_x(x_before, i_in);
			cov f_x_final;
			Matrix<scalar_type, m, process_noise_dof> F_w = f_w(x_, i_in);
			Matrix<scalar_type, n, process_noise_dof> f_w_final;

			for (std::vector<std::pair<std::pair<int, int>, int>>::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++)
			{
				int idx = (*it).first.first;
				int dim = (*it).first.second;
				int dof = (*it).second;
				for (int i = 0; i < n; i++)
				{
					for (int j = 0; j < dof; j++)
					{
						f_x_final(idx + j, i) = F_x2(dim + j, i);
					}
				}
				for (int i = 0; i < process_noise_dof; i++)
				{
					for (int j = 0; j < dof; j++)
					{
						f_w_final(idx + j, i) = F_w(dim + j, i);
					}
				}
			}

			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
			{
				int idx = (*it).first;
				int dim = (*it).second;
				for (int i = 0; i < 3; i++)
				{
					seg_SO3(i) = -1 * f_(dim + i) * dt;
				}
				MTK::SO3<scalar_type> res;
				res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1 / 2));
				F_x1.template block<3, 3>(idx, idx) = res.toRotationMatrix();
				res_temp_SO3 = MTK::A_matrix(seg_SO3);
				for (int i = 0; i < n; i++)
				{
					f_x_final.template block<3, 1>(idx, i) = res_temp_SO3 * (F_x2.template block<3, 1>(dim, i));
				}
				for (int i = 0; i < process_noise_dof; i++)
				{
					f_w_final.template block<3, 1>(idx, i) = res_temp_SO3 * (F_w.template block<3, 1>(dim, i));
				}
			}

			Matrix<scalar_type, 2, 3> res_temp_S2;
			MTK::vect<3, scalar_type> seg_S2;
			for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
			{
				int idx = (*it).first;
				int dim = (*it).second;
				for (int i = 0; i < 3; i++)
				{
					seg_S2(i) = f_(dim + i) * dt;
				}
				MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
				MTK::SO3<scalar_type> res;
				res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1 / 2));
				Eigen::Matrix<scalar_type, 2, 3> Nx;
				Eigen::Matrix<scalar_type, 3, 2> Mx;
				x_.S2_Nx_yy(Nx, idx);
				x_before.S2_Mx(Mx, vec, idx);
				F_x1.template block<2, 2>(idx, idx) = Nx * res.toRotationMatrix() * Mx;

				Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
				x_before.S2_hat(x_before_hat, idx);
				res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat * MTK::A_matrix(seg_S2).transpose();

				for (int i = 0; i < n; i++)
				{
					f_x_final.template block<2, 1>(idx, i) = res_temp_S2 * (F_x2.template block<3, 1>(dim, i));
				}
				for (int i = 0; i < process_noise_dof; i++)
				{
					f_w_final.template block<2, 1>(idx, i) = res_temp_S2 * (F_w.template block<3, 1>(dim, i));
				}
			}

			F_x1 += f_x_final * dt;
			P_ = F_x1 * P_ * F_x1.transpose() + (dt * f_w_final) * Q * (dt * f_w_final).transpose();
		}

		// iterated error state EKF update modified for one specific system. for fastlio2
		void update_iterated_fastlio2(bool &measure_valid, bool &iter_converge)
		{
			fastlio_datastruct<scalar_type> dyn_share;
			dyn_share.valid = true;
			dyn_share.converge = true;
			int t = 0;
			state x_propagated = x_;
			cov P_propagated = P_;
			int dof_Measurement;

			Matrix<scalar_type, n, 1> Kz;
			Matrix<scalar_type, n, n> KH;

			vectorized_state Jx_new = vectorized_state::Zero();
			for (int i = -1; i < maximum_iter; i++)
			{
				dyn_share.valid = true;
				// 1.计算residual和测量jacob矩阵
				h_dyn_share(x_, dyn_share);

				if (!dyn_share.valid)
				{
					measure_valid = false;
					continue;
				}

				double R = dyn_share.R;
				Eigen::Matrix<scalar_type, Eigen::Dynamic, 12> h_x = dyn_share.H;
				dof_Measurement = h_x.rows();
				// 2.计算当前误差
				vectorized_state error_x;
				x_.boxminus(error_x, x_propagated);
				Jx_new = error_x;

				P_ = P_propagated;

				// 3.计算特殊流形状态的jacob^-1 * error_x, 并对P进行预测更新 P = JPJ(用block子矩阵来计算，可以加速求解)
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
				{
					int idx = (*it).first;
					int dim = (*it).second;
					for (int i = 0; i < 3; i++)
					{
						seg_SO3(i) = error_x(idx + i);
					}

					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					Jx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * Jx_new.template block<3, 1>(idx, 0);
					for (int i = 0; i < n; i++)
					{
						P_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
					}
					for (int i = 0; i < n; i++)
					{
						P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				Matrix<scalar_type, 2, 2> res_temp_S2;
				MTK::vect<2, scalar_type> seg_S2;
				for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
				{
					int idx = (*it).first;
					int dim = (*it).second;
					for (int i = 0; i < 2; i++)
					{
						seg_S2(i) = error_x(idx + i);
					}

					Eigen::Matrix<scalar_type, 2, 3> Nx;
					Eigen::Matrix<scalar_type, 3, 2> Mx;
					x_.S2_Nx_yy(Nx, idx);
					x_propagated.S2_Mx(Mx, seg_S2, idx);
					res_temp_S2 = Nx * Mx;
					Jx_new.template block<2, 1>(idx, 0) = res_temp_S2 * Jx_new.template block<2, 1>(idx, 0);
					for (int i = 0; i < n; i++)
					{
						P_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
					}
					for (int i = 0; i < n; i++)
					{
						P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
					}
				}

				// 4.通过观测维度，区分使用哪种方式计算K
				if (n > dof_Measurement)
				{
					// K = PH^T(HPH^T+R)^-1
					Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> H_x = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
					H_x.topLeftCorner(dof_Measurement, 12) = h_x;
					auto K = P_ * H_x.transpose() * (H_x * P_ * H_x.transpose() + R * Eigen::Matrix<scalar_type, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement)).inverse();
					Kz = K * dyn_share.z;
					KH = K * H_x;
				}
				else
				{
					// K = (H^T R^-1 H + P^-1)^-1 H^T R^-1
					cov P_temp = (P_ / R).inverse();
					Eigen::Matrix<scalar_type, 12, 12> HTH = h_x.transpose() * h_x;
					P_temp.template block<12, 12>(0, 0) += HTH;
					cov P_inv = P_temp.inverse();
					Kz = P_inv.template block<n, 12>(0, 0) * h_x.transpose() * dyn_share.z;
					KH.setZero();
					KH.template block<n, 12>(0, 0) = P_inv.template block<n, 12>(0, 0) * HTH;
				}

				Matrix<scalar_type, n, 1> dx_ = Kz + (KH - Matrix<scalar_type, n, n>::Identity()) * Jx_new;
				// 5.更新状态
				x_.boxplus(dx_);
				dyn_share.converge = true;
				for (int i = 0; i < n; i++)
				{
					if (std::fabs(dx_[i]) > limit[i])
					{
						dyn_share.converge = false;
						break;
					}
				}
				iter_converge = iter_converge || dyn_share.converge;
#if 0
				printf("error state = (%d | %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f)\n",
					   dyn_share.converge, dx_[0], dx_[1], dx_[2], dx_[3], dx_[4], dx_[5], dx_[6], dx_[7], dx_[8], dx_[9], dx_[10],
					   dx_[11], dx_[12], dx_[13], dx_[14], dx_[15], dx_[16], dx_[17], dx_[18], dx_[19], dx_[20], dx_[21], dx_[22]);
#endif
				if (dyn_share.converge)
					t++;

				if (!t && i == maximum_iter - 2)
				{
					dyn_share.converge = true;
				}

				if (t > 1 || i == maximum_iter - 1)
				{
					L_ = P_;
					// 6.像3一样，最后传播一次P
					Matrix<scalar_type, 3, 3> res_temp_SO3;
					MTK::vect<3, scalar_type> seg_SO3;
					for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
					{
						int idx = (*it).first;
						for (int i = 0; i < 3; i++)
						{
							seg_SO3(i) = dx_(i + idx);
						}
						res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
						for (int i = 0; i < n; i++)
						{
							L_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
						}
						for (int i = 0; i < std::min((int)n, dof_Measurement); i++)
						{
							KH.template block<3, 1>(idx, i) = res_temp_SO3 * (KH.template block<3, 1>(idx, i));
						}
						for (int i = 0; i < n; i++)
						{
							L_.template block<1, 3>(i, idx) = (L_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
							P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						}
					}

					Matrix<scalar_type, 2, 2> res_temp_S2;
					MTK::vect<2, scalar_type> seg_S2;
					for (std::vector<std::pair<int, int>>::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++)
					{
						int idx = (*it).first;

						for (int i = 0; i < 2; i++)
						{
							seg_S2(i) = dx_(i + idx);
						}

						Eigen::Matrix<scalar_type, 2, 3> Nx;
						Eigen::Matrix<scalar_type, 3, 2> Mx;
						x_.S2_Nx_yy(Nx, idx);
						x_propagated.S2_Mx(Mx, seg_S2, idx);
						res_temp_S2 = Nx * Mx;
						for (int i = 0; i < n; i++)
						{
							L_.template block<2, 1>(idx, i) = res_temp_S2 * (P_.template block<2, 1>(idx, i));
						}
						for (int i = 0; i < std::min((int)n, dof_Measurement); i++)
						{
							KH.template block<2, 1>(idx, i) = res_temp_S2 * (KH.template block<2, 1>(idx, i));
						}
						for (int i = 0; i < n; i++)
						{
							L_.template block<1, 2>(i, idx) = (L_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
							P_.template block<1, 2>(i, idx) = (P_.template block<1, 2>(i, idx)) * res_temp_S2.transpose();
						}
					}

					// 7.状态方差的测量更新
					P_ = L_ - KH.template block<n, 12>(0, 0) * P_.template block<12, n>(0, 0);
					return;
				}
			}
		}

		// for pointlio
		void init_dyn_share_modified(processModel f_in, processMatrix1 f_x_in, measurementModel_dyn_share_modified h_dyn_share_in)
		{
			f = f_in;
			f_x = f_x_in;
			// f_w = f_w_in;
			h_dyn_share_modified_1 = h_dyn_share_in;
			maximum_iter = 1;
			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
			x_.build_SEN_state();
		}

		void init_dyn_share_modified_2h(processModel f_in, processMatrix1 f_x_in, measurementModel_dyn_share_modified h_dyn_share_in1, measurementModel_dyn_share_modified h_dyn_share_in2)
		{
			f = f_in;
			f_x = f_x_in;
			// f_w = f_w_in;
			h_dyn_share_modified_1 = h_dyn_share_in1;
			h_dyn_share_modified_2 = h_dyn_share_in2;
			maximum_iter = 1;
			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
			x_.build_SEN_state();
		}

		// iterated error state EKF propogation
		void predict(double &dt, processnoisecovariance &Q, const input &i_in, bool predict_state, bool prop_cov)
		{
			if (predict_state)
			{
				flatted_state f_ = f(x_, i_in);
				x_.oplus(f_, dt);
			}

			if (prop_cov)
			{
				flatted_state f_ = f(x_, i_in);
				cov_ f_x_ = f_x(x_, i_in);
				cov f_x_final;
				F_x1 = cov::Identity();
				for (std::vector<std::pair<std::pair<int, int>, int>>::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++)
				{
					int idx = (*it).first.first;
					int dim = (*it).first.second;
					int dof = (*it).second;
					for (int i = 0; i < n; i++)
					{
						for (int j = 0; j < dof; j++)
						{
							f_x_final(idx + j, i) = f_x_(dim + j, i);
						}
					}
				}

				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
				{
					int idx = (*it).first;
					int dim = (*it).second;
					for (int i = 0; i < 3; i++)
					{
						seg_SO3(i) = -1 * f_(dim + i) * dt;
					}
					MTK::SO3<scalar_type> res;
					res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1 / 2));
					F_x1.template block<3, 3>(idx, idx) = res.normalized().toRotationMatrix();
					res_temp_SO3 = MTK::A_matrix(seg_SO3);
					for (int i = 0; i < n; i++)
					{
						f_x_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_.template block<3, 1>(dim, i));
					}
				}

				F_x1 += f_x_final * dt;
				P_ = F_x1 * P_ * F_x1.transpose() + Q * (dt * dt);
			}
		}

		bool update_iterated_pointlio()
		{
			pointlio_datastruct<scalar_type> dyn_share;
			state x_propagated = x_;
			int dof_Measurement;
			double m_noise;
			for (int i = 0; i < maximum_iter; i++)
			{
				dyn_share.valid = true;
				h_dyn_share_modified_1(x_, dyn_share);
				if (!dyn_share.valid)
				{
					return false;
					// continue;
				}
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
				dof_Measurement = h_x.rows();
				m_noise = dyn_share.M_Noise;

				Matrix<scalar_type, n, Eigen::Dynamic> PHT;
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> HPHT;
				Matrix<scalar_type, n, Eigen::Dynamic> K;
				// if(n > dof_Measurement)
				{
					PHT = P_.template block<n, 12>(0, 0) * h_x.transpose();
					HPHT = h_x * PHT.topRows(12);
					for (int m = 0; m < dof_Measurement; m++)
					{
						HPHT(m, m) += m_noise;
					}
					K = PHT * HPHT.inverse();
				}
				Matrix<scalar_type, n, 1> dx_ = K * z;

				x_.boxplus(dx_);
				dyn_share.converge = true;

				P_ = P_ - K * h_x * P_.template block<12, n>(0, 0);
			}
			return true;
		}

		void update_iterated_dyn_share_IMU()
		{
			pointlio_datastruct<scalar_type> dyn_share;
			for (int i = 0; i < maximum_iter; i++)
			{
				dyn_share.valid = true;
				h_dyn_share_modified_2(x_, dyn_share);

				Matrix<scalar_type, 6, 1> z = dyn_share.z_IMU;

				Matrix<double, 30, 6> PHT;
				Matrix<double, 6, 30> HP;
				Matrix<double, 6, 6> HPHT;
				PHT.setZero();
				HP.setZero();
				HPHT.setZero();
				for (int i = 0; i < 6; i++)
				{
					if (!dyn_share.satu_check[i])
					{
						PHT.col(i) = P_.col(15 + i) + P_.col(24 + i);
						HP.row(i) = P_.row(15 + i) + P_.row(24 + i);
					}
				}
				for (int i = 0; i < 6; i++)
				{
					if (!dyn_share.satu_check[i])
					{
						HPHT.col(i) = HP.col(15 + i) + HP.col(24 + i);
					}
					HPHT(i, i) += dyn_share.R_IMU(i); //, l);
				}
				Eigen::Matrix<double, 30, 6> K = PHT * HPHT.inverse();

				Matrix<scalar_type, n, 1> dx_ = K * z;

				P_ -= K * HP;
				x_.boxplus(dx_);
			}
			return;
		}

		void change_x(state &input_state)
		{
			x_ = input_state;

			if ((!x_.vect_state.size()) && (!x_.SO3_state.size()) && (!x_.S2_state.size()) && (!x_.SEN_state.size()))
			{
				x_.build_S2_state();
				x_.build_SO3_state();
				x_.build_vect_state();
				x_.build_SEN_state();
			}
		}

		void change_P(cov &input_cov)
		{
			P_ = input_cov;
		}

		const state &get_x() const
		{
			return x_;
		}
		const cov &get_P() const
		{
			return P_;
		}
		state x_;

	private:
		cov P_;
		cov F_x1 = cov::Identity();
		cov F_x2 = cov::Identity();
		cov L_ = cov::Identity();

		processModel *f;
		processMatrix1 *f_x;
		processMatrix2 *f_w;

		measurementMatrix1 *h_x;

		measurementModel_dyn_share h_dyn_share;
		measurementModel_dyn_share_modified h_dyn_share_modified_1;
		measurementModel_dyn_share_modified h_dyn_share_modified_2;

		int maximum_iter = 0;
		scalar_type limit[n];

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
