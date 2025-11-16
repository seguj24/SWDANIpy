#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Empirical relation for calculating density and other properties of Earth's crust.

Reference:
    Brocher, T. M. (2005). 
    Empirical Relations between Elastic Wavespeeds and Density in the Earth's Crust.
    Bulletin of the Seismological Society of America, 95(6), 2081–2092.
"""

import numpy as np

def vs2vp(vs, method='brocher'):
    """
    Convert Vs to Vp.
    method='brocher' : Brocher (2006)
    method='gardner' : Gardner-style linear relation

    VS: between 0 and 4.5 km/s
    """
    if method == 'brocher':
        vp = 0.9409 + 2.094*vs - 0.8206*vs**2 + 0.2683*vs**3 - 0.0251*vs**4
    elif method == 'gardner':
        # Gardner's rule (approximation)
        vp = 1.16 * vs * 1.36
    else:
        raise ValueError("method must be 'brocher' or 'gardner'.")
    return vp


def vp2rho(vp, method='brocher'):
    """
    Convert Vp to density.
    method='brocher' : Brocher (2006)
    method='gardner' : Gardner's rule
    """
    if method == 'brocher':
        # Brocher (2006)
        return 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5
    elif method == 'gardner':
        # Gardner's rule
        return 1.74 * (vp**0.25)
    else:
        raise ValueError("method must be 'brocher' or 'gardner'.")


def rho2vp(rho):
    vp = 39.128*rho - 63.064*rho**2 + 37.083*rho**3 - 9.1819*rho**4 + 0.8228*rho**5
    return vp

def vp2vs(vp, line_type='normal'):
    if line_type == 'normal':
        """
        Brocher's regression fit
        VP: between 1.5 to 8 km/s
        """
        vs = 0.7858 - 1.2344*vp + 0.7949*vp**2 - 0.1238*vp**3 + 0.0064*vp**4
    return vs
    # if line_type == 'mica':
    # not yet
    #     """
    #     VP: between 5.25 and 7.25 km/s
    #     """
    #     vs = 2.88 + 0.52(vp-5.25)
        
def vp2poi(vp):
    """
    VP: between 1.5 and 8.5 km/s
    """
    poi = 0.8835 - 0.315*vp + 0.0491*vp**2 - 0.0024*vp**3
    return poi






# class conv_params():
#     """
#     Empirical relation for calculating density and other properties of Earth's crust.
    
#     Reference:
#         Brocher, T. M. (2005). 
#         Empirical Relations between Elastic Wavespeeds and Density in the Earth's Crust.
#         Bulletin of the Seismological Society of America, 95(6), 2081–2092.
    
#     Description:
#         This class calculates the VP, VS, Density, and Poisson's ratio of Earth's crust based on the input
#         using empirical relations.
        
#         Note: Calculation of density from the other parameters uses basic seismic wave equation.
        
#         Density will be calculated with VP and VS using simple seismic wave equation.
    
#     Attributes:
#         in_type (str): Input parameter type. Options are:
#             - 'vp': P-wave velocity (m/s).
#             - 'vs': S-wave velocity (m/s).
#             - 'rho': Density (kg/m³).
#             - 'poi': Poisson's ratio (unitless).
#         input_data (numpy.ndarray): Input data array.:
#             - vp (numpy.ndarray or None): Calculated P-wave velocity (m/s).
#             - vs (numpy.ndarray or None): Calculated S-wave velocity (m/s).
#             - rho (numpy.ndarray or None): Calculated density (kg/m³).
#             - poi (numpy.ndarray or None): Calculated Poisson's ratio (unitless).
        
#         output_data (dictionart): Outpot data dictionary. Keys are:
#             - 'vp': P-wave velocity (m/s).
#             - 'vs': S-wave velocity (m/s).
#             - 'rho': Density (kg/m³).
#             - 'poi': Poisson's ratio (unitless).
#     """
   
#     def __init__(self, in_type, input_data):
#         """
#         Initialize the empirical relation with input data.

#         Parameters:
#         ----------
#         in_type : str
#             Type of input parameter. Valid options are:
#                 - 'vp': P-wave velocity (m/s).
#                 - 'vs': S-wave velocity (m/s).
#                 - 'rho': Density (kg/m³).
#                 - 'poi': Poisson's ratio (unitless).
#         input_data : numpy.ndarray
#             Array containing the input parameter values.

#         Attributes Initialized:
#         -----------------------
#         vp : numpy.ndarray or None
#             P-wave velocity (m/s) (computed if not provided).
#         vs : numpy.ndarray or None
#             S-wave velocity (m/s) (computed if not provided).
#         rho : numpy.ndarray or None
#             Density (kg/m³) (computed if not provided).
#         poi : numpy.ndarray or None
#             Poisson's ratio (unitless) (computed if not provided).
#         """
#         self.in_type = in_type
#         self.input_data = input_data
#         self.vp = None
#         self.vs = None
#         self.rho = None
#         self.poi = None
        
#         self._compute_properties()


#     def get_properties(self):
#         """
#         Return all calculated properties as a dictionary.
#         """
#         return {
#             'rho': self.rho,
#             'vp': self.vp,
#             'vs': self.vs,
#             'poi': self.poi
#             }

#     def _compute_properties(self):
#         """
#         Compute all relevant properties based on the input type.
#         """
#         if self.in_type == 'rho':
#             self.rho = self.input_data
#             self.vp = self.rho2vp()
#             self.vs = self.vp2vs()
#             self.poi = self.vp2poi()
            
#         elif self.in_type == 'vp':
#             self.vp = self.input_data
#             self.vs = self.vp2vs()
#             # self.rho = self._rho_from_vp_and_vs()
#             self.rho = np.full(self.vp.shape, np.nan)
#             self.poi = self.vp2poi()
#             # print("Density from Vp is not supported then will be provided as 'nan'")
            
#         elif self.in_type == 'vs':
#             self.vs = self.input_data
#             self.vp = self.vs2vp()
#             # self.rho = self._rho_from_vp_and_vs()
#             self.rho = np.full(self.vs.shape, np.nan)
#             self.poi = self.vp2poi()
#             # print("Density from Vs is not supported then will be provided as 'nan'")
            
#         elif self.in_type == 'poi':
#             raise ValueError("Calculation from Poisson's ratio is not supported.")
#         else:
#             raise ValueError(f"Unknown input type: {self.in_type}")

    # def _rho_from_vp_and_vs(self):
    #     #### 음.... 이게 ....???...
    #     vp = self.vp
    #     vs = self.vs
    #     rho = (vs**2)/(vp**2 - 2*vs**2)
        
    #     return rho
    
    
    # def _vp_from_rho(self):
    #     """
    #     rho: between 2 and 3.5 g/cm^3
    #     """
    #     rho = self.rho
    #     vp = 39.128*rho - 63.064*rho**2 + 37.083*rho**3 - 9.1819*rho**4 + 0.8228*rho**5
    #     return vp
    
    # def _vs_from_vp(self, line_type='normal'):
    #     vp = self.vp
    #     if line_type == 'normal':
    #         """
    #         Brocher's regression fit
    #         VP: between 1.5 to 8 km/s
    #         """
    #         vs = 0.7858 - 1.2344*vp + 0.7949*vp**2 - 0.1238*vp**3 + 0.0064*vp**4
        
    #     # if line_type == 'mica':
    #     # not yet
    #     #     """
    #     #     VP: between 5.25 and 7.25 km/s
    #     #     """
    #     #     vs = 2.88 + 0.52(vp-5.25)
        
    #     return vs
    
    # def _vp_from_vs(self):
    #     """
    #     VS: between 0 and 4.5 km/s
    #     """
    #     vs = self.vs
    #     vp = 0.9409 + 2.094*vs - 0.8206*vs**2 + 0.2683*vs**3 - 0.0251*vs**4
        
    #     return vp
    
        
    # def _poi_from_vp(self):
    #     """
    #     VP: between 1.5 and 8.5 km/s
    #     """
    #     vp = self.vp
    #     poi = 0.8835 - 0.315*vp + 0.0491*vp**2 - 0.0024*vp**3
        
    #     return poi
    
    # def _rho_from_vp_and_vs(self):
    #     #### 음.... 이게 ....???...
    #     vp = self.vp
    #     vs = self.vs
    #     rho = (vs**2)/(vp**2 - 2*vs**2)
    #     return rho
    
    
    
# class Emvwv():
#     """
#     Elastic modulus VS. Wave Velocity
    
#     Reference:
#         Lianyang, 2017
#         Engineering properties of rocks
#     """
#     def __init__():
#         """
#         """






























    
