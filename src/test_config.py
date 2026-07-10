import config

# Test Last N
def test_last_n():
    assert config.do_trainable_layers("last_5",28) == [23,24,25,26,27]

# Test explicit list
# IMP - shouldnt we make "x,y,z" --> [x,y,z] for UX which takes spec as [x,y,z] and returns itself
def test_list() :
    assert config.do_trainable_layers("24,25,26,27",28) == [24,25,26,27]

def test_guardrail():
        try:
            config.do_trainable_layers("last_40", 28)
            assert False, "should have raised"
        except ValueError:
            pass

if __name__ == "__main__":
        test_last_n()
        test_list()
        test_guardrail()
        print("All config tests passed!")
