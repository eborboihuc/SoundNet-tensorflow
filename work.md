

### When read with torchfile, can use instance._obj to fetch dict
[Link](https://github.com/bshillingford/python-torchfile/blob/master/torchfile.py#L107)

```python
mydict = o['modules'][0]._obj
```

### Set value to variable and make it trainable
[Link](https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19_trainable.py#L135)

```python
def get_var(self, initial_value, name, idx, var_name):
    if self.data_dict is not None and name in self.data_dict:
        value = self.data_dict[name][idx]
    else:
        value = initial_value

    if self.trainable:
        var = tf.Variable(value, name=var_name)
    else:
        var = tf.constant(value, dtype=tf.float32, name=var_name)

    self.var_dict[(name, idx)] = var

    # print var_name, var.get_shape().as_list()
    assert var.get_shape() == initial_value.get_shape()

    return var
```


