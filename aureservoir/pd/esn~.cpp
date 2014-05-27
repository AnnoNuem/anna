/***************************************************************************/
/*!
 *  \file   esn~.cpp
 *
 *  \brief  Pure Data binding for the ESN class
 *
 *  \author Georg Holzmann, grh _at_ mur _dot_ at
 *  \date   Sept 2007
 *
 *   ::::_aureservoir_::::
 *   C++ library for analog reservoir computing neural networks
 *
 *   This library is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU Lesser General Public
 *   License as published by the Free Software Foundation; either
 *   version 2.1 of the License, or (at your option) any later version.
 *
 ***************************************************************************/

#include "m_pd.h"
#include "aureservoir/aureservoir.h"

using namespace aureservoir;

static t_class *esn_tilde_class;

typedef struct _esn_tilde {
  t_object  x_obj;
  t_sample f;
  ESN *net;
} t_esn_tilde;

static t_int *esn_tilde_perform(t_int *w)
{
  t_esn_tilde *x = (t_esn_tilde *)(w[1]);
  t_sample   *in =    (t_sample *)(w[2]);
  t_sample  *out =    (t_sample *)(w[3]);
  int          n =           (int)(w[4]);

  /// \todo dieses viele Kopieren vermeiden !!!

  DEVector netin(n), netout(n);

  for(int i=1; i<=n; ++i)
    netin(i) = in[i-1];

  netout = x->net->run( netin );

  for(int i=1; i<=n; ++i)
    out[i-1] = netout(i);

  return (w+5);
}

static void esn_tilde_dsp(t_esn_tilde *x, t_signal **sp)
{
  dsp_add(esn_tilde_perform, 4, x,
          sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
}

static void *esn_tilde_new(t_floatarg f)
{
  t_esn_tilde *x = (t_esn_tilde *)pd_new(esn_tilde_class);

  outlet_new(&x->x_obj, &s_signal);

  // init net
  x->net = new ESN;
  x->net->setReservoirSize(100);
  x->net->setReservoirParams(0.1, 0.8);
  x->net->init();

  // train net
  int train_size = 150;
  DEVector in(train_size), out(train_size);
  for(int i=1; i<=train_size; ++i)
  {
    in(i) = randval()*2-1;
    out(i) = randval()*2-1;
  }
  x->net->train(in, out, 10);

  return (void *)x;
}

static void esn_tilde_free(t_esn_tilde *x)
{
  delete x->net;
}

extern "C" {

void esn_tilde_setup(void)
{
  esn_tilde_class = class_new(gensym("esn~"),
        (t_newmethod)esn_tilde_new,
        (t_method)esn_tilde_free, sizeof(t_esn_tilde),
        CLASS_DEFAULT, A_DEFFLOAT, 0);

  // without A_DEFFLOAT it doesn't work !?
  class_addmethod(esn_tilde_class, (t_method)esn_tilde_dsp,
                  gensym("dsp"), A_DEFFLOAT, 0);

  CLASS_MAINSIGNALIN(esn_tilde_class, t_esn_tilde, f);
}

} // end extern "C"
