import numpy as np

class Focal_Smoothed_Hinge:
    def __init__(
        self,
        gamma_indct: float = 2.0,
        alpha_indct: float = 0.25,
        use_safe_hessian: bool = True,
        eps: float = 1e-9,
    ):
        """
        Parameters
        ----------
        gamma_indct      : Focal 的 γ (>0)
        alpha_indct      : Focal 的 α (0-1 之间)
        use_safe_hessian : True → Hessian 取 w·𝟙(tz<1)（更稳更快）
        eps              : 避免零 / 负 Hessian 的极小正数
        """
        self.gamma_indct = float(gamma_indct)
        self.alpha_indct = float(alpha_indct)
        self.use_safe_hessian = bool(use_safe_hessian)
        self.eps = float(eps)

    # ------------------------------------------------------------------ #
    #                    数值稳定辅助函数（sigmoid / 幂）                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """数值安全版 σ(z)"""
        return np.where(
            x >= 0.0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    @staticmethod
    def _robust_pow(x: np.ndarray, p: float) -> np.ndarray:
        """|x|**p · sign(x) —— 允许负底数配合非整数指数"""
        return np.sign(x) * np.power(np.abs(x), p)

    def focal_smoothed_hinge(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Scikit-Learn API 期望的 (y_true, y_pred) 签名  
        返回：grad, hess (均为 float32, C-contiguous)
        """
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)

        # ----- 1. 基础量 -------------------------------------------------
        sig   = self._sigmoid(y_pred)                   # σ(z)
        t     = y_true * 2.0 - 1.0                      # ±1
        pt    = y_true * sig + (1.0 - y_true) * (1.0 - sig)
        at    = self.alpha_indct * y_true + (1.0 - self.alpha_indct) * (1.0 - y_true)

        one_m_pt = np.clip(1.0 - pt, self.eps, 1.0 - self.eps)
        w        = at * self._robust_pow(one_m_pt, self.gamma_indct)

        # dw/dz
        dpt_dz   = sig * (1.0 - sig) * t
        dw_dz    = -self.gamma_indct * at * self._robust_pow(
            one_m_pt, self.gamma_indct - 1.0
        ) * dpt_dz

        # ----- 2. Smoothed-Hinge ----------------------------------------
        u        = t * y_pred
        mask     = (u < 1.0).astype(np.float32)
        margin   = 1.0 - u

        H        = 0.5 * margin**2 * mask              # H(z)（仅用于组合）
        dH_dz    = -margin * mask * t                  # ∂H/∂z
        d2H_dz2  = mask                                 # ∂²H/∂z²

        # ----- 3. 梯度 ---------------------------------------------------
        grad = w * dH_dz + dw_dz * H

        # ----- 4. Hessian -----------------------------------------------
        if self.use_safe_hessian:
            hess = w * d2H_dz2
        else:
            d2pt_dz2 = (
                sig * (1.0 - sig) * (1.0 - 2.0 * sig) * t
            )
            d2w_dz2 = (
                self.gamma_indct
                * (self.gamma_indct - 1.0)
                * at
                * self._robust_pow(one_m_pt, self.gamma_indct - 2.0)
                * dpt_dz**2
                - self.gamma_indct
                * at
                * self._robust_pow(one_m_pt, self.gamma_indct - 1.0)
                * d2pt_dz2
            )
            hess = w * d2H_dz2 + d2w_dz2 * H + 2.0 * dw_dz * dH_dz

        hess = np.clip(hess, self.eps, None)

        return grad.astype(np.float32), hess.astype(np.float32)

    # 允许把实例本身直接当作 objective=
    __call__ = focal_smoothed_hinge