"""
Bindings for Apple CGL.
"""
from __future__ import annotations
import builtins
import ctypes as ctypes
from ctypes import c_int as GLint
from ctypes import c_void_p as CGLContextObj
from ctypes import c_void_p as CGLPixelFormatObj
import enum as enum
import typing
__all__: list[str] = ['CGLContextObj', 'CGLError', 'CGLOpenGLProfile', 'CGLPixelFormatAttribute', 'CGLPixelFormatObj', 'CGLReleaseContext', 'CGLReleasePixelFormat', 'GLint', 'ctypes', 'enum']
class CGLOpenGLProfile(enum.IntEnum):
    CGLOGLPVersion_3_2_Core: typing.ClassVar[CGLOpenGLProfile]
    CGLOGLPVersion_GL3_Core: typing.ClassVar[CGLOpenGLProfile]
    CGLOGLPVersion_GL4_Core: typing.ClassVar[CGLOpenGLProfile]
    CGLOGLPVersion_Legacy: typing.ClassVar[CGLOpenGLProfile]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['CGLOGLPVersion_Legacy', 'CGLOGLPVersion_3_2_Core', 'CGLOGLPVersion_GL4_Core']
    _unhashable_values_: typing.ClassVar[list] = list()
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
                Generate the next value when not given.
        
                name: the name of the member
                start: the initial start value or None
                count: the number of existing members
                last_values: the list of values assigned
                
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class CGLPixelFormatAttribute(enum.IntEnum):
    """
    CGLPixelFormatAttribute enum values.
    """
    CGLPFAAccelerated: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAAcceleratedCompute: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAAccumSize: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAAllRenderers: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAAllowOfflineRenderers: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAAlphaSize: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAAuxBuffers: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAAuxDepthStencil: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFABackingStore: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFABackingVolatile: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAClosestPolicy: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAColorFloat: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAColorSize: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFACompliant: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFADepthSize: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFADisplayMask: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFADoubleBuffer: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAFullScreen: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAMPSafe: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAMaximumPolicy: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAMinimumPolicy: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAMultiScreen: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAMultisample: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFANoRecovery: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAOffScreen: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAOpenGLProfile: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAPBuffer: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFARemotePBuffer: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFARendererID: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFARobust: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFASample: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFASampleAlpha: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFASampleBuffers: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFASingleRenderer: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAStencilSize: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAStereo: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFASupersample: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFASupportsAutomaticGraphicsSwitching: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFATripleBuffer: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAVirtualScreenCount: typing.ClassVar[CGLPixelFormatAttribute]
    CGLPFAWindow: typing.ClassVar[CGLPixelFormatAttribute]
    _member_map_: typing.ClassVar[dict]
    _member_names_: typing.ClassVar[list] = ['CGLPFAAllRenderers', 'CGLPFATripleBuffer', 'CGLPFADoubleBuffer', 'CGLPFAColorSize', 'CGLPFAAlphaSize', 'CGLPFADepthSize', 'CGLPFAStencilSize', 'CGLPFAMinimumPolicy', 'CGLPFAMaximumPolicy', 'CGLPFASampleBuffers', 'CGLPFASample', 'CGLPFAColorFloat', 'CGLPFAMultisample', 'CGLPFASupersample', 'CGLPFASampleAlpha', 'CGLPFARendererID', 'CGLPFANoRecovery', 'CGLPFAAccelerated', 'CGLPFAClosestPolicy', 'CGLPFABackingStore', 'CGLPFABackingVolatile', 'CGLPFADisplayMask', 'CGLPFAAllowOfflineRenderers', 'CGLPFAAcceleratedCompute', 'CGLPFAOpenGLProfile', 'CGLPFASupportsAutomaticGraphicsSwitching', 'CGLPFAVirtualScreenCount', 'CGLPFAAuxBuffers', 'CGLPFAAccumSize', 'CGLPFAAuxDepthStencil', 'CGLPFAStereo', 'CGLPFAOffScreen', 'CGLPFAWindow', 'CGLPFACompliant', 'CGLPFAPBuffer', 'CGLPFARemotePBuffer', 'CGLPFASingleRenderer', 'CGLPFARobust', 'CGLPFAMPSafe', 'CGLPFAMultiScreen', 'CGLPFAFullScreen']
    _unhashable_values_: typing.ClassVar[list] = list()
    _use_args_: typing.ClassVar[bool] = True
    _value2member_map_: typing.ClassVar[dict]
    _member_type_ = builtins.int
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
                Generate the next value when not given.
        
                name: the name of the member
                start: the initial start value or None
                count: the number of existing members
                last_values: the list of values assigned
                
        """
    @staticmethod
    def _new_member_(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def _value_repr_(self):
        """
        Return repr(self).
        """
class CGLError(RuntimeError):
    pass
def _make_checked(func):
    ...
CGLReleaseContext: ctypes.CDLL.__init__.<locals>._FuncPtr
CGLReleasePixelFormat: ctypes.CDLL.__init__.<locals>._FuncPtr
_CGL: ctypes.CDLL
_CGLChoosePixelFormat: ctypes.CDLL.__init__.<locals>._FuncPtr
_CGLCreateContext: ctypes.CDLL.__init__.<locals>._FuncPtr
_CGLErrorString: ctypes.CDLL.__init__.<locals>._FuncPtr
_CGLLockContext: ctypes.CDLL.__init__.<locals>._FuncPtr
_CGLReleaseContext: ctypes.CDLL.__init__.<locals>._FuncPtr
_CGLReleasePixelFormat: ctypes.CDLL.__init__.<locals>._FuncPtr
_CGLSetCurrentContext: ctypes.CDLL.__init__.<locals>._FuncPtr
_CGLUnlockContext: ctypes.CDLL.__init__.<locals>._FuncPtr
