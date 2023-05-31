import inspect
import wrapt

class Foo: pass
class Bar(Foo, wrapt.ObjectProxy): pass
inspect.getattr_static(Bar({}), 'bar')
